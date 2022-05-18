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
calculateConnectionFrequencyBasedOnNodeSentenceSubgraphs = True	#else calculateFrequencyBasedOnNodes (flat)	#more advanced method of node similarity comparison #compares similarity of subgraphs of sentence nodes (rather than similarity of just the nodes themselves)

calculateReferenceSimilarityUsingWordVectorSimilarity = False	#else calculateReferenceSimilarityUsingIdenticalConceptsLemmasInGraph
calculateReferenceSimilarityBasedOnNodeSentenceSubgraphs = True	#else calculateReferenceSimilarityBasedOnNodes (flat)

headNodePosTags = ["CC", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]

graphNodeTypeUnknown = 0
graphNodeTypeNormal = 1
graphNodeTypeHead = 2

conceptID = 0	#special instance ID for concepts
maxTimeDiff = 999999
minRecency = 0.1	#CHECKTHIS: requires calibration (> 0: ensures recency for unencountered concepts is not zero - required for metric)
maxTimeDiffForMatchingInstance = 2	#time in sentence index diff	#CHECKTHIS: requires calibration
metricThresholdToCreateConnection = 1.0	#CHECKTHIS: requires calibration
metricThresholdToCreateReference = 1.0	#CHECKTHIS: requires calibration

graphNodeDictionary = {}	#dict indexed by lemma, every entry is a dictionary of GraphNode instances indexed by instanceID (first instance is special; reserved for concept)
#graphConnectionsDictionary = {}	#dict indexed tuples (lemma1, instanceID1, lemma2, instanceID2), every entry is a tuple of GraphNode instances/concepts (instanceNode1, instanceNode2) [directionality: 1=source, 2=target]
	#this is used for visualisation/fast lookup purposes only - can trace node graphNodeTargets/graphNodeSources instead

class GraphNode:
	def __init__(self, instanceID, word, lemma, wordVector, posTag, nodeGraphType, activationTime, w):
		self.instanceID = instanceID
		self.word = word
		self.lemma = lemma
		self.wordVector = wordVector	#numpy array
		self.posTag = posTag	#nlp in context prediction only (not certain)
		self.graphNodeType = nodeGraphType
		self.activationTime = activationTime	#used to calculate recency
		self.wTemp = w #temporary sentence word index (used for reference resolution only)
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
	#sentenceGraphNodeDictionary = {}	#local/isolated/temporary graph of sentence instance nodes (before reference resolution)
		
	tokenisedSentence = tokeniseSentence(sentence)
	sentenceLength = len(tokenisedSentence)
	
	
	#declare graph nodes;
	for w, token in enumerate(tokenisedSentence):	

		word = getTokenWord(token)
		lemma = getTokenLemma(token)
		wordVector = getTokenWordVector(token)	#numpy word vector
		posTag = getTokenPOStag(token)
		activationTime = calculateActivationTime(sentenceIndex)
		nodeGraphType = getNodeGraphType(posTag)
		
		#add concept to dictionary (if non-existent);
		if lemma not in graphNodeDictionary:
			instanceID = conceptID
			conceptNode = GraphNode(instanceID, word, lemma, wordVector, posTag, nodeGraphType, currentTime, w)
			addInstanceNodeToGraph(lemma, instanceID, conceptNode)

		#add instance to local/temporary sentenceNodeList (reference resolution is required before adding nodes to graph);
		instanceIDprelim = getNewInstanceID(lemma)	#same instance id will be assigned to identical lemmas in sentence (which is not approprate in the case they refer to independent instances) - will be reassign instance id after reference resolution
		instanceNode = GraphNode(instanceIDprelim, word, lemma, wordVector, posTag, nodeGraphType, currentTime, w)
		print("create new instanceNode; ", instanceNode.lemma, ": instanceID=", instanceNode.instanceID)

		sentenceNodeList.append(instanceNode)

	#add connections to local/isolated/temporary graph (syntactical tree);
	foundConnectionOutgoing = [False]*sentenceLength	#found connection (direction: source to target)
	foundConnectionIncoming = [False]*sentenceLength	#found connection (direction: source to target)
	for distance in range(1, sentenceLength):	#search for proximal connections before distal connections
		for w in range(sentenceLength):	
			w2 = w + distance
			connectionDirection = True	
			if(w2 < sentenceLength):
				if((not foundConnectionOutgoing[w]) and (not foundConnectionIncoming[w2])):
					node1 = sentenceNodeList[w]
					node2 = sentenceNodeList[w2]

					proximity = calculateProximity(w, w2)
					frequency = calculateFrequencyConnection(sentenceNodeList, node1, node2)
					recency = calculateRecencyConnection(sentenceNodeList, node1, node2, currentTime)
					metric = calculateMetricConnection(proximity, frequency, recency)
					
					if(metric > metricThresholdToCreateConnection):
						print("create connection; w w2 = ", w, " ", w2, ", node1.lemma node2.lemma = ", node1.lemma, " ", node2.lemma, ", metric = ", metric)
						connectionDirection = True	#CHECKTHIS: always assume left to right directionality
						foundConnectionOutgoing[w] = True
						foundConnectionIncoming[w2] = True
						createGraphConnectionWrapper(node1, node2, connectionDirection, addToConnectionsDictionary=False)
						

	#peform reference resolution after building syntactical tree (any instance of successful reference identification will insert syntactical tree into graph)		
	#resolve references		
	resolvedReferences = [False]*sentenceLength
	for w in range(sentenceLength):	
		node1 = sentenceNodeList[w]
		foundReference, referenceNode, maxSimilarity = findMostSimilarReferenceInGraph(sentenceNodeList, node1, currentTime)
		print("findMostSimilarReferenceInGraph maxSimilarity = ", maxSimilarity)
		if(foundReference and (maxSimilarity > metricThresholdToCreateReference)):
			print("replaceReference")
			replaceReference(sentenceNodeList, node1, referenceNode, currentTime)
		else:
			instanceID = getNewInstanceID(lemma)
			addInstanceNodeToGraph(node1.lemma, instanceID, node1)
	
	#limitations (CHECKTHIS):
	#- currently only support 1 connection (per direction) per word in sentence
	#- infers directionality (source/target) of connection based on w1/w2 word order


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
	newInstanceID = len(graphNodeDictionary[lemma])
	return newInstanceID

def calculateMetricReference(similarity, recency):
	#print("\tsimilarity = ", similarity)
	#print("\trecency = ", recency)
	metric = similarity*recency #CHECKTHIS: requires calibration - normalisation of factors is required
	#print("\tmetric = ", metric)
	return metric
		
def calculateMetricConnection(proximity, frequency, recency):
	#print("\tproximity = ", proximity)
	#print("\tfrequency = ", frequency)
	#print("\trecency = ", recency)
	metric = proximity*frequency*recency #CHECKTHIS: requires calibration - normalisation of factors is required
	#print("\tmetric = ", metric)
	return metric
	
def calculateActivationTime(sentenceIndex):
	activationTime = sentenceIndex
	return activationTime

def calculateProximity(w, w2):
	proximity = abs(w-w2)
	return proximity
	
def calculateFrequencyConnection(sentenceNodeList, node1, node2):
	#CHECKTHIS; note compares node subgraph source components (not target components)
	frequency = compareNodeConnectionSimilarity(sentenceNodeList, node1, node2)	 #CHECKTHIS requires update - currently uses rudimentary word vector similarity comparison
	return frequency


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
						

def calculateRecencyConnection(sentenceNodeList, node1, node2, currentTime):
	foundMostRecentIdenticalConnection, mostRecentConnectionTimeDiff = findMostRecentIdenticalConnection(sentenceNodeList, node1, node2, currentTime)
	recency = calculateRecency(mostRecentConnectionTimeDiff)
	return recency
	
def findMostRecentIdenticalConnection(sentenceNodeList, node1, node2, currentTime):
	foundMostRecentIdenticalConnection = False
	#mostRecentInstanceNode = None
	mostRecentConnectionTimeDiff = maxTimeDiff
	node1ConceptInstances = graphNodeDictionary[node1.lemma]
	for instanceID1, instanceNode1 in node1ConceptInstances.items():
		if(instanceNode1.activationTime != currentTime):	#ignore instances that were added from same sentence	#OR: instanceNode1 is not in(sentenceNodeList)
			node2ConceptInstances = graphNodeDictionary[node2.lemma]
			for instanceID2, instanceNode2 in node2ConceptInstances.items():
				if(instanceNode2.activationTime != currentTime):	#ignore instances that were added from same sentence	#OR: instanceNode2 is not in(sentenceNodeList)
					for instanceID1target, instanceNode1target in instanceNode1.graphNodeTargets.items():
						if(instanceNode1target.lemma == node2.lemma):
							connectionTime = min(instanceNode1.activationTime, instanceNode2.activationTime)
							connectionTimeDiff = calculateTimeDiffAbsolute(mostRecentConnectionTime, currentTime)
							if(connectionTimeDiff < mostRecentConnectionTimeDiff):
								mostRecentConnectionTimeDiff = connectionTimeDiff
								foundMostRecentIdenticalConnection = True
						
	return foundMostRecentIdenticalConnection, mostRecentConnectionTimeDiff
		
#not currently used;
def findMostRecentInstance(lemma, currentTime):	
	#print("findMostRecentInstance") 
	#calculates recency of most recent instance, and returns this instance
	foundMostRecentInstanceNode = False
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
	recency = 1.0/timeDiff
	return recency
	


def createGraphConnectionWrapper(node1, node2, connectionDirection, addToConnectionsDictionary=True):
	if(connectionDirection):
		createGraphConnection(graph, node1, node2)
	else:
		createGraphConnection(graph, node2, node1)
		
def createGraphConnection(graph, nodeSource, nodeTarget, addToConnectionsDictionary):
	addInstanceNodeToDictionary(nodeSource.graphNodeTargets, nodeTarget.lemma, nodeTarget.instanceID, nodeTarget)
	addInstanceNodeToDictionary(nodeTarget.graphNodeSources, nodeSource.lemma, nodeSource.instanceID, nodeSource)

	#if(addToConnectionsDictionary):
	#	graphConnectionKey = createGraphConnectionKey(nodeSource, nodeTarget)
	#	graphConnectionsDictionary[graphConnectionKey] = (nodeSource, nodeTarget)

#def createGraphConnectionKey(nodeSource, nodeTarget):
#	connectionKey = (nodeSource.lemma, nodeSource.instanceID, nodeTarget.lemma, nodeTarget.instanceID)
#	return connectionKey

#def identifyNodeType(node):


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
		if(subgraphNode2 in sentenceNodeList):	#verify subgraph instance was referenced in current sentence
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
	
				
def compareNodeConnectionSimilarity(sentenceNodeList, node1, node2):
	if(calculateConnectionFrequencyUsingWordVectorSimilarity):
		similarity = compareNodeWordVectorSimilarity(sentenceNodeList, node1, node2)		#compareNodeCorpusAssociationFrequency
	else:
		print("compareNodeCorpusAssociationFrequency currently requires calculateConnectionFrequencyUsingWordVectorSimilarity - no alternate method coded")
		exit()
	return similarity

def compareNodeWordVectorSimilarity(sentenceNodeList, node1, node2):
	if(calculateConnectionFrequencyBasedOnNodeSentenceSubgraphs):
		subgraphArtificalWordVector1 = node1.wordVector
		subgraphArtificalWordVector2 = node2.wordVector
		subgraphSize1 = calculateSubgraphArtificialWordVector(sentenceNodeList, node1, subgraphArtificalWordVector1, 1)
		subgraphSize2 = calculateSubgraphArtificialWordVector(sentenceNodeList, node2, subgraphArtificalWordVector2, 1)
		subgraphArtificalWordVector1 = np.divide(subgraphArtificalWordVector1, float(subgraphSize1))
		subgraphArtificalWordVector2 = np.divide(subgraphArtificalWordVector2, float(subgraphSize2))
		similarity = calculateWordVectorSimilarity(subgraphArtificalWordVector1, subgraphArtificalWordVector2)		
	else:
		similarity = calculateWordVectorSimilarity(node1.wordVector, node2.wordVector)
	return similarity

def calculateSubgraphArtificialWordVector(sentenceNodeList, node, subgraphArtificalWordVector, subgraphSize):
	#CHECKTHIS: requires update - currently uses rudimentary combined word vector similarity comparison
	for subgraphInstanceID, subgraphNode in node.graphNodeSources.items():	
		if(subgraphNode in sentenceNodeList):	#verify subgraph instance was referenced in current sentence
			subgraphNodeWordVector = subgraphNode.wordVector
			subgraphArtificalWordVector = np.add(subgraphArtificalWordVector, subgraphNodeWordVector)
			subgraphSize = calculateSubgraphArtificialWordVector(sentenceNodeList, subgraphNode, subgraphArtificalWordVector, subgraphSize+1)
	return subgraphSize
	
def calculateWordVectorSimilarity(wordVector1, wordVector2):
	wordVectorDiff = compareWordVectors(wordVector1, wordVector2)
	similarity = 1.0 - wordVectorDiff
	#print("similarity = ", similarity)
	return similarity

def compareWordVectors(wordVector1, wordVector2):
	wordVectorDiff = np.mean(np.absolute(np.subtract(wordVector1, wordVector2)))
	#print("wordVectorDiff = ", wordVectorDiff)
	return wordVectorDiff



	
def getNodeGraphType(posTag):
	#determines if node should be visualised as a head (verb/action, conjunction, etc)
	graphNodeType = graphNodeTypeNormal
	if(posTag in headNodePosTags):
		graphNodeType = graphNodeTypeHead
	return graphNodeType

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
	
		
