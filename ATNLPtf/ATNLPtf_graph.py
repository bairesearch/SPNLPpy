"""ATNLPtf_graph.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATNLPtf_main.py

# Usage:
see ATNLPtf_main.py

# Description:
ATNLP graph - generate syntactical/semantic tree/graph using input vectors (based on word proximity, frequency, and recency)

"""

import numpy as np
import spacy
spacyWordVectorGenerator = spacy.load('en_core_web_md')	#spacy.load('en_core_web_lg')
import ANNtf2_loadDataset
import ATNLPtf_getAllPossiblePosTags

#calculateFrequencyConnection method: calculates frequency of co-occurance of words/subsentences in corpus 
calculateConnectionFrequencyUsingWordVectorSimilarity = True		#CHECKTHIS requires update - currently uses rudimentary word vector similarity comparison
#CHECKTHIS; if subgraph is not symmetrical, will incorrectly weigh dynamic wordvector/recency calculations
calculateConnectionFrequencyBasedOnNodeSentenceSubgraphs = False	#optional (not required to compare hidden node similarity as artificial word vectors are generated for new hidden nodes in sentence)	#else calculateWordVectorSimilarity (flat)	#compares similarity of subgraphs of sentence nodes (rather than similarity of just the nodes themselves)
calculateConnectionRecencyBasedOnNodeSentenceSubgraphs = False	#optional (not required to compare hidden node concept recency as artificial concept recency values are generated for new hidden nodes in sentence)		#else compareNodeRecency (flat)

calculateReferenceSimilarityUsingWordVectorSimilarity = False	#else calculateReferenceSimilarityUsingIdenticalConceptsLemmasInGraph
calculateReferenceSimilarityBasedOnNodeSentenceSubgraphs = True	#else calculateReferenceSimilarityBasedOnNodes (flat)

addConceptNodesToDatabase = True	#add concept to dictionary (if non-existent) - not currently used
#contiguousNodesGraph = True 	#mandatory: nodes must be contiguous (word order)


#referenceSetDelimiterNodePosTags = ["CC", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]

graphNodeTypeUnknown = 0
graphNodeTypeLeaf = 1	#lemma
graphNodeTypeHidden = 2	#branch
graphNodeTypeHead = 3	#tree

conceptID = 0	#special instance ID for concepts
maxTimeDiff = 10.0	#calculateTimeDiff(minRecency)	#CHECKTHIS: requires calibration (<= ~10: ensures timeDiff for unencountered concepts is not infinite - required for metric)	#units: sentenceIndex
#minRecency = calculateRecency(maxTimeDiff)	#  minRecency = 0.1	#CHECKTHIS: requires calibration (> 0: ensures recency for unencountered concepts is not zero - required for metric)	
maxRecency = 1.0
maxTimeDiffForMatchingInstance = 2	#time in sentence index diff	#CHECKTHIS: requires calibration
metricThresholdToCreateConnection = 1.0	#CHECKTHIS: requires calibration
metricThresholdToCreateReference = 1.0	#CHECKTHIS: requires calibration

graphNodeDictionary = {}	#dict indexed by lemma, every entry is a dictionary of GraphNode instances indexed by instanceID (first instance is special; reserved for concept)
#graphConnectionsDictionary = {}	#dict indexed tuples (lemma1, instanceID1, lemma2, instanceID2), every entry is a tuple of GraphNode instances/concepts (instanceNode1, instanceNode2) [directionality: 1=source, 2=target]
	#this is used for visualisation/fast lookup purposes only - can trace node graphNodeTargets/graphNodeSources instead

class GraphNode:
	def __init__(self, instanceID, word, lemma, wordVector, conceptRecency, posTag, nodeGraphType, activationTime, w, wMin, wMax):
		self.instanceID = instanceID
		self.word = word
		self.lemma = lemma
		self.wordVector = wordVector	#numpy array
		self.conceptRecencySentenceTreeArtificial = conceptRecency
		self.posTag = posTag	#nlp in context prediction only (not certain)
		self.graphNodeType = nodeGraphType
		self.activationTime = activationTime	#used to calculate recency
		self.w = w #temporary sentence word index (used for reference resolution only)
		self.wMin = wMin	#temporary sentence word index (used for reference resolution only) - min of all hidden nodes
		self.wMax = wMax	#temporary sentence word index (used for reference resolution only) - max of all hidden nodes
		self.referenceSentenceTreeArtificial = False	#temporary flag: node has been reference by current sentence (used for reference resolution only)
		#self.activationLevel = 0.0	#used to calculate recency
		#self.frequency = None	#float
		#connections;
		self.graphNodeTargets = {}	#dict indexed by lemma, every entry is a dictionary of GraphNode instances indexed by instanceID 
		self.graphNodeSources = {}	#dict indexed by lemma, every entry is a dictionary of GraphNode instances indexed by instanceID
		#self.foundRecentIndex = False	#temporary var (indicates referencing a previously declared instance in the article)
		
	
def generateGraph(sentenceIndex, sentence):
	
	print("\n\ngenerateGraph: sentenceIndex = ", sentenceIndex, "; ", sentence)
	
	currentTime = calculateActivationTime(sentenceIndex)
	
	sentenceNodeList = []	#local/temporary list of sentence instance nodes (before reference resolution)
	#sentenceLeafNodeList = []	#local/temporary list of sentence instance nodes (before reference resolution)
	connectivityStackNodeList = []	#temporary list of nodes on connectivity stack
	#sentenceGraphNodeDictionary = {}	#local/isolated/temporary graph of sentence instance nodes (before reference resolution)
		
	tokenisedSentence = tokeniseSentence(sentence)
	sentenceLength = len(tokenisedSentence)
	
	#declare graph nodes;
	for w, token in enumerate(tokenisedSentence):	

		word = getTokenWord(token)
		lemma = getTokenLemma(token)
		wordVector = getTokenWordVector(token)	#numpy word vector
		conceptRecency = calculateConceptRecencyLeafNode(sentenceNodeList, lemma, currentTime)	#units: min time diff (not recency metric)
		posTag = getTokenPOStag(token)
		activationTime = calculateActivationTime(sentenceIndex)
		nodeGraphType = graphNodeTypeLeaf
		
		if(addConceptNodesToDatabase):
			#add concept to dictionary (if non-existent) - not currently used;
			if lemma not in graphNodeDictionary:
				instanceID = conceptID
				conceptNode = GraphNode(instanceID, word, lemma, wordVector, conceptRecency, posTag, nodeGraphType, currentTime, w, w, w)
				addInstanceNodeToGraph(lemma, instanceID, conceptNode)

		#add instance to local/temporary sentenceNodeList (reference resolution is required before adding nodes to graph);
		instanceIDprelim = getNewInstanceID(lemma)	#same instance id will be assigned to identical lemmas in sentence (which is not approprate in the case they refer to independent instances) - will be reassign instance id after reference resolution
		instanceNode = GraphNode(instanceIDprelim, word, lemma, wordVector, conceptRecency, posTag, nodeGraphType, currentTime, w, w, w)
		print("create new instanceNode; ", instanceNode.lemma, ": instanceID=", instanceNode.instanceID)

		sentenceNodeList.append(instanceNode)
		#sentenceLeafNodeList.append(instanceNode)
		connectivityStackNodeList.append(instanceNode)

	#add connections to local/isolated/temporary graph (syntactical tree);
	if(sentenceLength > 1):
		headNodeFound = False
		while not headNodeFound:

			connectionNode1 = None
			connectionNode2 = None
			maxConnectionMetric = 0.0

			connectionFound = False
			
			for node1StackIndex, node1 in enumerate(connectivityStackNodeList):
				for node2StackIndex, node2 in enumerate(connectivityStackNodeList):
					if(node1StackIndex != node2StackIndex):
						#print("node1.wMax = ", node1.wMax)
						#print("node2.wMin = ", node2.wMin)
						if(node1.wMax+1 == node2.wMin):
							proximity = calculateProximity(node1.w, node2.w)
							frequency = calculateFrequencyConnection(sentenceNodeList, node1, node2)
							recency = calculateRecencyConnection(sentenceNodeList, node1, node2, currentTime)	#minimise the difference in recency between left/right node
							connectionMetric = calculateMetricConnection(proximity, frequency, recency)
							if(connectionMetric > maxConnectionMetric):
								#print("connectionMetric found")
								connectionFound = True
								maxConnectionMetric = connectionMetric
								connectionNode1 = node1
								connectionNode2 = node2
			
			if(not connectionFound):
				print("error connectionFound - check calculateMetricConnection parameters > 0.0, maxConnectionMetric = ", maxConnectionMetric)
				exit()
				
			print("create connection; w1 w2 = ", connectionNode1.w, " ", connectionNode2.w, ", connectionNode1.lemma connectionNode2.lemma = ", connectionNode1.lemma, " ", connectionNode2.lemma, ", metric = ", maxConnectionMetric)

			#CHECKTHIS limitation - infers directionality (source/target) of connection based on w1/w2 word order		
			connectionDirection = True	#CHECKTHIS: always assume left to right directionality

			word = connectionNode1.word + connectionNode2.word
			lemma = connectionNode1.lemma + connectionNode2.lemma
			wordVector = np.mean([connectionNode1.wordVector, connectionNode2.wordVector])	#CHECKTHIS; if subgraph is not symmetrical, will incorrectly weigh word vectors
			conceptRecency = average(connectionNode1.conceptRecencySentenceTreeArtificial, connectionNode2.conceptRecencySentenceTreeArtificial) #CHECKTHIS; if subgraph is not symmetrical, will incorrectly weigh recency
			posTag = None
			activationTime = average(connectionNode1.activationTime, connectionNode2.activationTime) 		#calculateActivationTime(sentenceIndex)
			nodeGraphType = graphNodeTypeHidden
			w = average(connectionNode1.w, connectionNode2.w)
			wMin = min(connectionNode1.wMin, connectionNode2.wMin)
			wMax = max(connectionNode1.wMax, connectionNode2.wMax)
			
			instanceIDprelim = getNewInstanceID(lemma)
			hiddenNode = GraphNode(instanceIDprelim, word, lemma, wordVector, conceptRecency, posTag, nodeGraphType, currentTime, w, wMin, wMax)
			createGraphConnectionWrapper(hiddenNode, connectionNode1, connectionNode2, connectionDirection, addToConnectionsDictionary=False)

			connectivityStackNodeList.remove(connectionNode1)
			connectivityStackNodeList.remove(connectionNode2)
			connectivityStackNodeList.append(hiddenNode)

			if(len(connectivityStackNodeList) == 1):
				headNodeFound = True
				hiddenNode.graphNodeType = graphNodeTypeHead	#reference set delimiter (captures primary subject/action/object of sentence clause)
	else:
		 node1 = connectivityStackNodeList[0]
		 node1.graphNodeType = graphNodeTypeHead

		
	#peform reference resolution after building syntactical tree (any instance of successful reference identification will insert syntactical tree into graph)		
	#resolve references		
	resolvedReferences = [False]*sentenceLength
	for node1 in sentenceNodeList:	
		if(not node1.referenceSentenceTreeArtificial):
			foundReference, referenceNode, maxSimilarity = findMostSimilarReferenceInGraph(sentenceNodeList, node1, currentTime)
			print("findMostSimilarReferenceInGraph maxSimilarity = ", maxSimilarity)
			if(foundReference and (maxSimilarity > metricThresholdToCreateReference)):
				print("replaceReference")
				#CHECKTHIS limitation; only replaces highest level node in reference set - consider replacing all nodes in reference set
				replaceReference(sentenceNodeList, node1, referenceNode, currentTime)
			else:
				#instanceID = getNewInstanceID(lemma)
				addInstanceNodeToGraph(node1.lemma, node1.instanceID, node1)
	


def average(number1, number2):
	return (number1 + number2) / 2.0
  
def createSubDictionaryForConcept(dic, lemma):
	dic[lemma] = {}	#create empty dictionary for new concept
			
def findInstanceNodeInGraph(lemma, instanceID):
	node = graphNodeDictionary[lemma][instanceID]
	return node

def addInstanceNodeToGraph(lemma, instanceID, node):
	addInstanceNodeToDictionary(graphNodeDictionary, lemma, instanceID, node)

def addInstanceNodeToDictionary(dic, lemma, instanceID, node):
	if lemma not in dic:
		createSubDictionaryForConcept(dic, lemma)
	dic[lemma][instanceID] = node
	
def getNewInstanceID(lemma):
	if lemma in graphNodeDictionary:
		newInstanceID = len(graphNodeDictionary[lemma])
	else:
		newInstanceID = 0
	return newInstanceID

def calculateMetricReference(similarity, recency):
	#print("\tsimilarity = ", similarity)
	#print("\trecency = ", recency)
	metric = similarity*recency #CHECKTHIS: requires calibration - normalisation of factors is required
	#print("\tmetric = ", metric)
	return metric
		
def calculateMetricConnection(proximity, frequency, recency):
	metric = proximity*frequency*recency #CHECKTHIS: requires calibration - normalisation of factors is required
	#print("\tproximity = ", proximity)
	#print("\tfrequency = ", frequency)
	#print("\trecency = ", recency)
	#print("\t\tmetric = ", metric)
	return metric
	
def calculateActivationTime(sentenceIndex):
	activationTime = sentenceIndex
	return activationTime

def calculateProximity(w, w2):
	proximity = abs(w-w2)
	return proximity



def replaceReference(sentenceNodeList, node1, referenceNode, currentTime):
	referenceNode.activationTime = currentTime
	
	for instanceID1souce, node1source in node1.graphNodeSources.items():
		for instanceID1souceTarget, node1sourceTarget in node1source.graphNodeTargets.items():
			if(node1sourceTarget == node1):
				node1source.graphNodeTargets[instanceID1souceTarget] = referenceNode	#replace target of previous word with reference node
	for instanceID1target, node1target in node1.graphNodeTargets.items():
		for instanceID1targetSource, node1targetSource in node1target.graphNodeSources.items():
			if(node1targetSource == node1):
				node1target.graphNodeSources[instanceID1targetSource] = referenceNode	#replace source of next word with reference node
						


#recency metric:
def calculateRecencyConnection(sentenceNodeList, node1, node2, currentTime):
	if(calculateConnectionRecencyBasedOnNodeSentenceSubgraphs):
		subgraphArtificalRecency1 = 0
		subgraphArtificalRecency2 = 0
		subgraphSize1 = calculateSubgraphArtificialRecency(sentenceNodeList, node1, subgraphArtificalRecency1, 0)
		subgraphSize2 = calculateSubgraphArtificialRecency(sentenceNodeList, node2, subgraphArtificalRecency2, 0)
		subgraphArtificalRecency1 = subgraphArtificalRecency1, subgraphSize1
		subgraphArtificalRecency2 = subgraphArtificalRecency2, subgraphSize2
		timeDiffConnection = compareNodeRecency(sentenceNodeList, node1, node2)
	else:
		timeDiffConnection = compareNodeRecency(sentenceNodeList, node1, node2)
	recencyDiffConnection = calculateRecency(timeDiffConnection)
	return recencyDiffConnection

def calculateSubgraphArtificialRecency(sentenceNodeList, node, subgraphArtificalRecency, subgraphSize):
	subgraphArtificalRecency = subgraphArtificalRecency + subgraphNode.conceptRecencySentenceTreeArtificial
	subgraphSize = subgraphSize + 1
	#CHECKTHIS: requires update - currently uses rudimentary combined minTimeDiff similarity comparison
	for subgraphInstanceID, subgraphNode in node.graphNodeSources.items():	
		if(subgraphNode in sentenceNodeList):	#verify subgraph instance was referenced in current sentence
			subgraphArtificalRecency, subgraphSize = calculateSubgraphArtificialRecency(sentenceNodeList, subgraphNode, subgraphArtificalRecency, subgraphSize)
	return subgraphArtificalRecency, subgraphSize
	
def compareNodeRecency(sentenceNodeList, node1, node2):
	timeDiffConnection = abs(node1.conceptRecencySentenceTreeArtificial - node2.conceptRecencySentenceTreeArtificial)
	return timeDiffConnection
	
def calculateSubgraphMostRecentIdenticalConnection(sentenceNodeList, node1, nodeToCompare, numberOfConnections1):
	#TODO: verify calculate source to target connections only
	numberOfConnections2 = calculateSubgraphNumberIdenticalConcepts2(sentenceNodeList, node1, nodeToCompare, 0)
	numberOfConnections1 += numberOfConnections2
	for subgraphInstanceID1, subgraphNode1 in node1.graphNodeSources.items():	
		if(subgraphNode1 in sentenceNodeList):	#verify subgraph instance was referenced in current sentence
			numberOfConnections1 = calculateSubgraphNumberIdenticalConcepts1(sentenceNodeList, subgraphNode1, nodeToCompare, numberOfConnections1)
	return numberOfConnections1	

def calculateConceptRecencyLeafNode(sentenceNodeList, lemma, currentTime):
	foundMostRecentInstanceNode, mostRecentInstanceNode, mostRecentInstanceTimeDiff = findMostRecentInstance(sentenceNodeList, lemma, currentTime)
	if(not mostRecentInstanceNode):
		mostRecentInstanceTimeDiff = maxTimeDiff
	return mostRecentInstanceTimeDiff
	
def findMostRecentInstance(sentenceNodeList, lemma, currentTime):	
	#print("findMostRecentInstance") 
	#calculates recency of most recent instance, and returns this instance
	foundMostRecentInstanceNode = False
	mostRecentInstanceNode = None
	mostRecentInstanceTimeDiff = None
	if(lemma in graphNodeDictionary):
		instanceDict2 = graphNodeDictionary[lemma]
		minTimeDiff = maxTimeDiff
		mostRecentInstanceNode = None
		for instanceID2, node2 in instanceDict2.items():
			if(node2.activationTime != currentTime):	#ignore instances that were added from same sentence	#OR: node2 is not in(sentenceNodeList)
				timeDiff = calculateTimeDiffAbsolute(node2.activationTime, currentTime)
				#print("timeDiff = ", timeDiff)
				if(timeDiff < minTimeDiff):
					foundMostRecentInstanceNode = True
					minTimeDiff = timeDiff
					mostRecentInstanceNode = node2	#dict key

		mostRecentInstanceTimeDiff = minTimeDiff
	#print("mostRecentInstanceTimeDiff = ", mostRecentInstanceTimeDiff)
	return foundMostRecentInstanceNode, mostRecentInstanceNode, mostRecentInstanceTimeDiff
	
def calculateRecencyAbsolute(node2activationTime, currentTime):
	timeDiff = calculateTimeDiffAbsolute(node2activationTime, currentTime)
	recency = calculateRecency(timeDiff)
	return recency

def calculateTimeDiffAbsolute(node2activationTime, currentTime):
	timeDiff = currentTime - node2activationTime
	return timeDiff
	
def calculateRecency(timeDiff):
	if(timeDiff == 0):
		recency = maxRecency	#1.0
	else:
		recency = 1.0/timeDiff
	return recency

def calculateTimeDiff(recency):
	timeDiff = 1.0/recency
	return timeDiff	


#connection:
def createGraphConnectionWrapper(hiddenNode, node1, node2, connectionDirection, addToConnectionsDictionary=True):
	if(connectionDirection):
		createGraphConnection(hiddenNode, node1, node2, addToConnectionsDictionary)
	else:
		createGraphConnection(hiddenNode, node2, node1, addToConnectionsDictionary)
		
def createGraphConnection(hiddenNode, node1, node2, addToConnectionsDictionary):
	addInstanceNodeToDictionary(node1.graphNodeTargets, hiddenNode.lemma, hiddenNode.instanceID, hiddenNode)
	addInstanceNodeToDictionary(node2.graphNodeTargets, hiddenNode.lemma, hiddenNode.instanceID, hiddenNode)
	addInstanceNodeToDictionary(hiddenNode.graphNodeSources, node1.lemma, node1.instanceID, node1)
	addInstanceNodeToDictionary(hiddenNode.graphNodeSources, node2.lemma, node2.instanceID, node2)

	#if(addToConnectionsDictionary):
	#	graphConnectionKey = createGraphConnectionKey(hiddenNode, node1, node2)
	#	graphConnectionsDictionary[graphConnectionKey] = (hiddenNode, node1, node2)

#def createGraphConnectionKey(hiddenNode, node1, node2):
#	connectionKey = (hiddenNode.lemma, hiddenNode.instanceID, node1.lemma, node1.instanceID, node2.lemma, node2.instanceID)
#	return connectionKey

#def identifyNodeType(node):


#reference similarity:
def findMostSimilarReferenceInGraph(sentenceNodeList, node1, currentTime):
	foundReference = False
	referenceNode = None
	maxSimilarity = 0

	node1ConceptInstances = graphNodeDictionary[node1.lemma]	#current limitation: only reference identical lemmas [future allow referencing based on word vector similarity]
	for instanceID1, instanceNode1 in node1ConceptInstances.items():
		if(instanceNode1.activationTime != currentTime):	#ignore instances that were added from same sentence	#OR: instanceNode1 is not in(sentenceNodeList)
			similarity = compareNodeReferenceSimilarity(sentenceNodeList, node1, instanceNode1)
			recency = calculateRecencyAbsolute(instanceNode1.activationTime, currentTime)
			metric = calculateMetricReference(similarity, recency)
			if(metric > maxSimilarity):
				maxSimilarity = metric
				referenceNode = instanceNode1
				foundReference = True
				
	return foundReference, referenceNode, maxSimilarity
	
def compareNodeReferenceSimilarity(sentenceNodeList, node1, node2):
	if(calculateReferenceSimilarityUsingWordVectorSimilarity):
		similarity = compareNodeWordVectorSimilarity(sentenceNodeList, node1, node2)
	else:
		similarity = compareNodeIdenticalConceptSimilarity(sentenceNodeList, node1, node2)
	#print("compareNodeReferenceSimilarity similarity = ", similarity)
	return similarity

def compareNodeIdenticalConceptSimilarity(sentenceNodeList, node1, node2):
	if(calculateReferenceSimilarityBasedOnNodeSentenceSubgraphs):
		similarity = calculateSubgraphNumberIdenticalConcepts1(sentenceNodeList, node1, node2, 0)			
	else:
		similarity = calculateNumberIdenticalConcepts(node1, node2)
	return similarity

#compares all nodes in node1 subgraph (to nodeToCompare subgraphs)
#recurse node1 subgraph
def calculateSubgraphNumberIdenticalConcepts1(sentenceNodeList, node1, nodeToCompare, numberOfConnections1):
	#TODO: verify calculate source to target connections only
	numberOfConnections2 = calculateSubgraphNumberIdenticalConcepts2(sentenceNodeList, node1, nodeToCompare, 0)
	numberOfConnections1 += numberOfConnections2
	for subgraphInstanceID1, subgraphNode1 in node1.graphNodeSources.items():	
		if(subgraphNode1 in sentenceNodeList):	#verify subgraph instance was referenced in current sentence
			numberOfConnections1 = calculateSubgraphNumberIdenticalConcepts1(sentenceNodeList, subgraphNode1, nodeToCompare, numberOfConnections1)
	return numberOfConnections1

#compares node1 with all nodes in node2 subgraph
#recurse node2 subgraph
def calculateSubgraphNumberIdenticalConcepts2(sentenceNodeList, node1, node2, numberOfConnections):
	#TODO: verify calculate source to target connections only
	if(node1.lemma == node2.lemma):
		numberOfConnections += 1
	for subgraphInstanceID2, subgraphNode2 in node2.graphNodeSources.items():	
		if(subgraphNode2 not in sentenceNodeList):	#verify subgraph instance was not referenced in current sentence
			numberOfConnections = calculateSubgraphNumberIdenticalConcepts2(sentenceNodeList, node1, subgraphNode2, numberOfConnections)
	return numberOfConnections
		
def calculateNumberIdenticalConcepts(node, nodeEnd):
	#TODO: verify calculate source to target connections only
	numberOfConnections = 0
	instanceDict1 = graphNodeDictionary[node.lemma]
	for instanceID1, node1 in instanceDict1.items():
		for instanceID2, node2 in node1.graphNodeTargets.items():
			if(node2.lemma == nodeEnd.lemma):
				numberOfConnections += 1
	return numberOfConnections
	

#connection similarity:
def calculateFrequencyConnection(sentenceNodeList, node1, node2):
	#CHECKTHIS; note compares node subgraph source components (not target components)
	frequency = compareNodeConnectionSimilarity(sentenceNodeList, node1, node2)	 #CHECKTHIS requires update - currently uses rudimentary word vector similarity comparison
	return frequency

def compareNodeConnectionSimilarity(sentenceNodeList, node1, node2):
	if(calculateConnectionFrequencyUsingWordVectorSimilarity):
		similarity = compareNodeWordVectorSimilarity(sentenceNodeList, node1, node2)		#compareNodeCorpusAssociationFrequency
	else:
		print("compareNodeCorpusAssociationFrequency currently requires calculateConnectionFrequencyUsingWordVectorSimilarity - no alternate method coded")
		exit()
	return similarity

def compareNodeWordVectorSimilarity(sentenceNodeList, node1, node2):
	if(calculateConnectionFrequencyBasedOnNodeSentenceSubgraphs):
		subgraphArtificalWordVector1 = np.zeros(np.shape(node1.wordVector))
		subgraphArtificalWordVector2 = np.zeros(np.shape(node2.wordVector))
		subgraphSize1 = calculateSubgraphArtificialWordVector(sentenceNodeList, node1, subgraphArtificalWordVector1, 0)
		subgraphSize2 = calculateSubgraphArtificialWordVector(sentenceNodeList, node2, subgraphArtificalWordVector2, 0)
		subgraphArtificalWordVector1 = np.divide(subgraphArtificalWordVector1, float(subgraphSize1))
		subgraphArtificalWordVector2 = np.divide(subgraphArtificalWordVector2, float(subgraphSize2))
		similarity = calculateWordVectorSimilarity(subgraphArtificalWordVector1, subgraphArtificalWordVector2)		
	else:
		similarity = calculateWordVectorSimilarity(node1.wordVector, node2.wordVector)
	return similarity

def calculateSubgraphArtificialWordVector(sentenceNodeList, node, subgraphArtificalWordVector, subgraphSize):
	#CHECKTHIS: requires update - currently uses rudimentary combined word vector similarity comparison
	subgraphArtificalWordVector = np.add(subgraphArtificalWordVector, node.wordVector)
	subgraphSize = subgraphSize + 1
	for subgraphInstanceID, subgraphNode in node.graphNodeSources.items():	
		if(subgraphNode in sentenceNodeList):	#verify subgraph instance was referenced in current sentence
			subgraphArtificalWordVector, subgraphSize = calculateSubgraphArtificialWordVector(sentenceNodeList, subgraphNode, subgraphArtificalWordVector, subgraphSize)
	return subgraphArtificalWordVector, subgraphSize
	
def calculateWordVectorSimilarity(wordVector1, wordVector2):
	wordVectorDiff = compareWordVectors(wordVector1, wordVector2)
	similarity = 1.0 - wordVectorDiff
	#print("similarity = ", similarity)
	return similarity

def compareWordVectors(wordVector1, wordVector2):
	wordVectorDiff = np.mean(np.absolute(np.subtract(wordVector1, wordVector2)))
	#print("wordVectorDiff = ", wordVectorDiff)
	return wordVectorDiff



	
#def getNodeGraphType(posTag):
#	#determines if node should be visualised as a head (verb/action, conjunction, etc)
#	graphNodeType = graphNodeTypeLeaf
#	#if(posTag in referenceSetDelimiterNodePosTags):
#	#	graphNodeType = graphNodeTypeHead
#	return graphNodeType

def tokeniseSentence(sentence):
	tokenList = spacyWordVectorGenerator(sentence)
	return tokenList

def getTokenWord(token):
	word = token.text
	return word
	
def getTokenLemma(token):
	lemma = token.lemma_
	return lemma
		
def getTokenWordVector(token):
	wordVector = token.vector	#cpu: type numpy
	return wordVector

def getTokenPOStag(token):
	#nlp in context prediction only (not certain)
	posTag = token.pos_
	return posTag
	
		
