"""ATNLPtf_syntacticalGraph.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATNLPtf_main.py

# Usage:
see ATNLPtf_main.py

# Description:
ATNLP Syntactical Graph - generate syntactical tree/graph using input vectors (based on word proximity, frequency, and recency heuristics)

"""

import numpy as np
import spacy
spacyWordVectorGenerator = spacy.load('en_core_web_md')	#spacy.load('en_core_web_lg')
import ANNtf2_loadDataset
from ATNLPtf_syntacticalNodeClass import *

performReferenceResolution = False	#temporarily disable to ensure single leaf node per word in sentence	#disable with generateSemanticGraph
	
drawSyntacticalGraph = True
drawSyntacticalGraphNodeColours = False
if(drawSyntacticalGraph):
	import ATNLPtf_syntacticalGraphDraw
	if(drawSyntacticalGraphNodeColours):
		from ATNLPtf_semanticNodeClass import identifyEntityType

calibrateConnectionMetricParameters = True
printVerbose = False

#calculateFrequencyConnection method: calculates frequency of co-occurance of words/subsentences in corpus 
calculateConnectionFrequencyUsingWordVectorSimilarity = True		#CHECKTHIS requires update - currently uses rudimentary word vector similarity comparison
#CHECKTHIS; if subgraph is not symmetrical, will incorrectly weigh non-dynamic (iterative) wordvector/recency calculations
calculateConnectionFrequencyBasedOnNodeSentenceSubgraphsDynamic = True	#optional (not required to compare hidden node similarity as artificial word vectors are generated for new hidden nodes in sentence)	#else calculateWordVectorSimilarity (flat)	#compares similarity of subgraphs of sentence nodes (rather than similarity of just the nodes themselves)
calculateConnectionRecencyBasedOnNodeSentenceSubgraphsDynamic = True	#optional (not required to compare hidden node concept recency as artificial concept recency values are generated for new hidden nodes in sentence)		#else compareNodeTime (flat)

calculateReferenceSimilarityUsingWordVectorSimilarity = False	#else calculateReferenceSimilarityUsingIdenticalConceptsLemmasInGraph
calculateReferenceSimilarityBasedOnNodeSentenceSubgraphsDynamic = True	#mandatory	#else calculateReferenceSimilarityBasedOnNodes (flat)

addConceptNodesToDatabase = True	#add concept to dictionary (if non-existent) - not currently used
#contiguousNodesGraph = True 	#mandatory: nodes must be contiguous (word order)


conceptID = 0	#special instance ID for concepts
maxTimeDiff = 10.0	#calculateTimeDiff(minRecency)	#CHECKTHIS: requires calibration (<= ~10: ensures timeDiff for unencountered concepts is not infinite - required for metric)	#units: sentenceIndex
#minRecency = calculateRecency(maxTimeDiff)	#  minRecency = 0.1	#CHECKTHIS: requires calibration (> 0: ensures recency for unencountered concepts is not zero - required for metric)	
maxRecency = 1.0
maxTimeDiffForMatchingInstance = 2	#time in sentence index diff	#CHECKTHIS: requires calibration
metricThresholdToCreateConnection = 1.0	#CHECKTHIS: requires calibration
metricThresholdToCreateReference = 1.0	#CHECKTHIS: requires calibration

graphNodeDictionary = {}	#dict indexed by lemma, every entry is a dictionary of SyntacticalNode instances indexed by instanceID (first instance is special; reserved for concept)
#graphConnectionsDictionary = {}	#dict indexed tuples (lemma1, instanceID1, lemma2, instanceID2), every entry is a tuple of SyntacticalNode instances/concepts (instanceNode1, instanceNode2) [directionality: 1=source, 2=target]
	#this is used for visualisation/fast lookup purposes only - can trace node graphNodeTargetList/graphNodeSourceList instead


def generateSyntacticalGraphStringInput(sentenceIndex, sentence):

	print("\n\ngenerateSyntacticalGraph: sentenceIndex = ", sentenceIndex, "; ", sentence)

	tokenisedSentence = tokeniseSentence(sentence)
	sentenceLength = len(tokenisedSentence)

	if(sentenceLength > 1):
		return generateSyntacticalGraph(sentenceIndex, tokenisedSentence)
	else:
		print("error: sentenceLength !> 1")
		exit()
			
def generateSyntacticalGraph(sentenceIndex, tokenisedSentence):

	ATNLPtf_syntacticalGraphDraw.setColourSyntacticalNodes(drawSyntacticalGraphNodeColours)
	print("ATNLPtf_syntacticalGraph: ATNLPtf_syntacticalGraphDraw.drawSyntacticalGraphNodeColours = ", ATNLPtf_syntacticalGraphDraw.drawSyntacticalGraphNodeColours)
	
	currentTime = calculateActivationTime(sentenceIndex)

	if(drawSyntacticalGraph):
		ATNLPtf_syntacticalGraphDraw.clearSyntacticalGraph()

	sentenceLeafNodeList = []	#local/temporary list of sentence instance nodes (before reference resolution)		
	sentenceTreeNodeList = []	#local/temporary list of sentence instance nodes (before reference resolution)
	connectivityStackNodeList = []	#temporary list of nodes on connectivity stack
	#sentenceGraphNodeDictionary = {}	#local/isolated/temporary graph of sentence instance nodes (before reference resolution)
		
	sentenceLength = len(tokenisedSentence)
	
	#declare graph nodes;
	for w, token in enumerate(tokenisedSentence):	

		word = getTokenWord(token)
		lemma = getTokenLemma(token)
		wordVector = getTokenWordVector(token)	#numpy word vector
		conceptTime = calculateConceptRecencyLeafNode(sentenceLeafNodeList, lemma, currentTime)	#units: min time diff (not recency metric)
		posTag = getTokenPOStag(token)
		activationTime = calculateActivationTime(sentenceIndex)
		nodeGraphType = graphNodeTypeLeaf
		treeLevel = 0

		if(addConceptNodesToDatabase):
			#add concept to dictionary (if non-existent) - not currently used;
			if lemma not in graphNodeDictionary:
				instanceID = conceptID
				conceptNode = SyntacticalNode(instanceID, word, lemma, wordVector, conceptTime, posTag, nodeGraphType, currentTime, w, w, w, treeLevel)
				addInstanceNodeToGraph(lemma, instanceID, conceptNode)

		#add instance to local/temporary sentenceLeafNodeList (reference resolution is required before adding nodes to graph);
		instanceIDprelim = getNewInstanceID(lemma)	#same instance id will be assigned to identical lemmas in sentence (which is not approprate in the case they refer to independent instances) - will be reassign instance id after reference resolution
		instanceNode = SyntacticalNode(instanceIDprelim, word, lemma, wordVector, conceptTime, posTag, nodeGraphType, currentTime, w, w, w, treeLevel)
		if(printVerbose):
			print("create new instanceNode; ", instanceNode.lemma, ": instanceID=", instanceNode.instanceID)

		sentenceLeafNodeList.append(instanceNode)
		sentenceTreeNodeList.append(instanceNode)
		connectivityStackNodeList.append(instanceNode)
		
		if(drawSyntacticalGraph):
			if(drawSyntacticalGraphNodeColours):	
				entityType = identifyEntityType(instanceNode)
				instanceNode.entityType = entityType
			ATNLPtf_syntacticalGraphDraw.drawSyntacticalGraphNode(instanceNode, w, treeLevel)

	#add connections to local/isolated/temporary graph (syntactical tree);
	if(calibrateConnectionMetricParameters):
		recencyList = []
		proximityList = []
		frequencyList = []	
		metricList = []

	headNodeFound = False
	graphHeadNode = None
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
						if(printVerbose):
							print("calculateMetricConnection: node1.lemma = ", node1.lemma, ", node2.lemma = ", node2.lemma)
						proximity = calculateProximityConnection(node1.w, node2.w)
						frequency = calculateFrequencyConnection(sentenceTreeNodeList, node1, node2)
						recency = calculateRecencyConnection(sentenceTreeNodeList, node1, node2, currentTime)	#minimise the difference in recency between left/right node
						connectionMetric = calculateMetricConnection(proximity, frequency, recency)
						if(connectionMetric > maxConnectionMetric):
							#print("connectionMetric found")
							connectionFound = True
							maxConnectionMetric = connectionMetric
							connectionNode1 = node1
							connectionNode2 = node2
						
						if(calibrateConnectionMetricParameters):
							proximityList.append(proximity)
							frequencyList.append(frequency)
							recencyList.append(recency)
							metricList.append(connectionMetric)
			
		if(not connectionFound):
			print("error connectionFound - check calculateMetricConnection parameters > 0.0, maxConnectionMetric = ", maxConnectionMetric)
			exit()

		if(printVerbose):
			print("create connection; w1 w2 = ", connectionNode1.w, " ", connectionNode2.w, ", connectionNode1.lemma connectionNode2.lemma = ", connectionNode1.lemma, " ", connectionNode2.lemma, ", metric = ", maxConnectionMetric)

		#CHECKTHIS limitation - infers directionality (source/target) of connection based on w1/w2 word order		
		connectionDirection = True	#CHECKTHIS: always assume left to right directionality

		word = connectionNode1.word + connectionNode2.word
		lemma = connectionNode1.lemma + connectionNode2.lemma
		if(calculateConnectionFrequencyBasedOnNodeSentenceSubgraphsDynamic):
			wordVector = None
		else:
			wordVector = np.mean([connectionNode1.wordVector, connectionNode2.wordVector])	#CHECKTHIS; if subgraph is not symmetrical, will incorrectly weigh word vectors
		if(calculateConnectionRecencyBasedOnNodeSentenceSubgraphsDynamic):
			conceptTime = None
		else:
			conceptTime = mean([connectionNode1.conceptTimeSentenceTreeArtificial, connectionNode2.conceptTimeSentenceTreeArtificial]) #CHECKTHIS; if subgraph is not symmetrical, will incorrectly weigh recency
		posTag = None
		activationTime = mean([connectionNode1.activationTime, connectionNode2.activationTime]) 		#calculateActivationTime(sentenceIndex)
		nodeGraphType = graphNodeTypeBranch
		treeLevel = max(connectionNode1.treeLevel, connectionNode2.treeLevel) + 1

		w = mean([connectionNode1.w, connectionNode2.w])
		wMin = min(connectionNode1.wMin, connectionNode2.wMin)
		wMax = max(connectionNode1.wMax, connectionNode2.wMax)

		instanceIDprelim = getNewInstanceID(lemma)
		hiddenNode = SyntacticalNode(instanceIDprelim, word, lemma, wordVector, conceptTime, posTag, nodeGraphType, currentTime, w, wMin, wMax, treeLevel)
		createGraphConnectionWrapper(hiddenNode, connectionNode1, connectionNode2, connectionDirection, addToConnectionsDictionary=False)

		sentenceTreeNodeList.append(hiddenNode)
		connectivityStackNodeList.remove(connectionNode1)
		connectivityStackNodeList.remove(connectionNode2)
		connectivityStackNodeList.append(hiddenNode)

		if(drawSyntacticalGraph):
			ATNLPtf_syntacticalGraphDraw.drawSyntacticalGraphNode(hiddenNode, w, treeLevel)
			ATNLPtf_syntacticalGraphDraw.drawSyntacticalGraphConnection(hiddenNode, connectionNode1)
			ATNLPtf_syntacticalGraphDraw.drawSyntacticalGraphConnection(hiddenNode, connectionNode2)

		if(len(connectivityStackNodeList) == 1):
			headNodeFound = True
			hiddenNode.graphNodeType = graphNodeTypeHead	#reference set delimiter (captures primary subject/action/object of sentence clause)
			graphHeadNode = hiddenNode

	if(calibrateConnectionMetricParameters):
		proximityMinMeanMax = minMeanMaxList(proximityList)
		frequencyMinMeanMax = minMeanMaxList(frequencyList)
		recencyMinMeanMax = minMeanMaxList(recencyList)
		metricMinMeanMax = minMeanMaxList(metricList)
		print("proximityMinMeanMax = ", proximityMinMeanMax)
		print("frequencyMinMeanMax = ", frequencyMinMeanMax)
		print("recencyMinMeanMax = ", recencyMinMeanMax)
		print("metricMinMeanMax = ", metricMinMeanMax)
			
	if(performReferenceResolution):
		#peform reference resolution after building syntactical tree (any instance of successful reference identification will insert syntactical tree into graph)		
		#resolve references		
		#CHECKTHIS limitation; only replaces highest level node in subgraph/reference set - consider replacing all nodes in subgraph/reference set
		resolvedReferences = [False]*sentenceLength
		for node1 in sentenceTreeNodeList:	
			if(not node1.referenceSentenceTreeArtificial):
				foundReference, referenceNode, maxSimilarity = findMostSimilarReferenceInGraph(sentenceTreeNodeList, node1, currentTime)
				if(foundReference and (maxSimilarity > metricThresholdToCreateReference)):
					print("replaceReference: findMostSimilarReferenceInGraph maxSimilarity = ", maxSimilarity)
					replaceReference(node1, referenceNode, currentTime)
				else:
					#instanceID = getNewInstanceID(lemma)
					addInstanceNodeToGraph(node1.lemma, node1.instanceID, node1)
	
	if(drawSyntacticalGraph):
		ATNLPtf_syntacticalGraphDraw.displaySyntacticalGraph()
		
	return sentenceLeafNodeList, sentenceTreeNodeList, graphHeadNode


def createSubDictionaryForConcept(dic, lemma):
	dic[lemma] = {}	#create empty dictionary for new concept
			
def findInstanceNodeInGraph(lemma, instanceID):
	node = graphNodeDictionary[lemma][instanceID]
	return node

#connection:
def createGraphConnectionWrapper(hiddenNode, node1, node2, connectionDirection, addToConnectionsDictionary=True):
	if(connectionDirection):
		createGraphConnection(hiddenNode, node1, node2, addToConnectionsDictionary)
	else:
		createGraphConnection(hiddenNode, node2, node1, addToConnectionsDictionary)
		
def createGraphConnection(hiddenNode, node1, node2, addToConnectionsDictionary):
	addConnectionToNodeTargets(node1, hiddenNode)
	addConnectionToNodeTargets(node2, hiddenNode)
	addConnectionToNodeSources(hiddenNode, node1)
	addConnectionToNodeSources(hiddenNode, node2)
	node1.sourceNodePosition = sourceNodePositionFirst
	node2.sourceNodePosition = sourceNodePositionSecond

	#if(addToConnectionsDictionary):
	#	graphConnectionKey = createGraphConnectionKey(hiddenNode, node1, node2)
	#	graphConnectionsDictionary[graphConnectionKey] = (hiddenNode, node1, node2)

#def createGraphConnectionKey(hiddenNode, node1, node2):
#	connectionKey = (hiddenNode.lemma, hiddenNode.instanceID, node1.lemma, node1.instanceID, node2.lemma, node2.instanceID)
#	return connectionKey




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
	#print("\t\tcalculateMetricConnection: metric = ", metric, "; proximity = ", proximity, ", frequency = ", frequency, ", recency = ", recency)
	return metric
	
def calculateActivationTime(sentenceIndex):
	activationTime = sentenceIndex
	return activationTime

def calculateProximityConnection(w, w2):
	proximity = 1.0 / abs(w-w2)	#CHECKTHIS: requires calibration
	#proximity = 1.0	#complete deweight of proximity parameter
	return proximity



def replaceReference(node1, referenceNode, currentTime):
	referenceNode.activationTime = currentTime
	
	for node1sourceIndex, node1source in enumerate(node1.graphNodeSourceList):
		for node1sourceTargetIndex, node1sourceTarget in enumerate(node1source.graphNodeTargetList):
			if(node1sourceTarget == node1):
				node1source.graphNodeTargetList[node1sourceTargetIndex] = referenceNode	#replace target of previous word with reference node
	for node1targetIndex, node1target in enumerate(node1.graphNodeTargetList):
		for node1targetSourceIndex, node1targetSource in enumerate(node1target.graphNodeSourceList):
			if(node1targetSource == node1):
				node1target.graphNodeSourceList[node1targetSourceIndex] = referenceNode	#replace source of next word with reference node
						


#recency metric:
def calculateRecencyConnection(sentenceTreeNodeList, node1, node2, currentTime):
	#CHECKTHIS: requires calibration
	#CHECKTHIS: requires update - currently uses rudimentary combined minTimeDiff similarity comparison
	if(calculateConnectionRecencyBasedOnNodeSentenceSubgraphsDynamic):
		subgraphArtificalTime1 = calculateSubgraphArtificialTime(sentenceTreeNodeList, node1)
		subgraphArtificalTime2 = calculateSubgraphArtificialTime(sentenceTreeNodeList, node2)
		timeDiffConnection = compareTime(subgraphArtificalTime1, subgraphArtificalTime2)
	else:
		timeDiffConnection = compareNodeTime(sentenceTreeNodeList, node1, node2)
	recencyDiffConnection = calculateRecency(timeDiffConnection)
	return recencyDiffConnection
	
def calculateSubgraphArtificialTime(sentenceTreeNodeList, node):
	subgraphArtificalTime = 0
	subgraphArtificalTime, subgraphSize = calculateSubgraphArtificialTimeRecurse(sentenceTreeNodeList, node, subgraphArtificalTime, 0)
	subgraphArtificalTime = (subgraphArtificalTime / subgraphSize)
	return subgraphArtificalTime

def calculateSubgraphArtificialTimeRecurse(sentenceTreeNodeList, node, subgraphArtificalTime, subgraphSize):
	if(node.graphNodeType == graphNodeTypeLeaf):
		subgraphArtificalTime = subgraphArtificalTime + node.conceptTimeSentenceTreeArtificial
		subgraphSize = subgraphSize + 1
	for subgraphNode in node.graphNodeSourceList:	
		if(subgraphNode in sentenceTreeNodeList):	#verify subgraph instance was referenced in current sentence
			subgraphArtificalTime, subgraphSize = calculateSubgraphArtificialTimeRecurse(sentenceTreeNodeList, subgraphNode, subgraphArtificalTime, subgraphSize)
	return subgraphArtificalTime, subgraphSize

def compareNodeTime(sentenceTreeNodeList, node1, node2):
	timeDiffConnection = compareTime(node1.conceptTimeSentenceTreeArtificial, node2.conceptTimeSentenceTreeArtificial)
	return timeDiffConnection

def compareTime(time1, time2):
	timeDiffConnection = abs(time1 - time2)
	return timeDiffConnection
	
	
def calculateSubgraphMostRecentIdenticalConnection(sentenceTreeNodeList, node1, nodeToCompare, numberOfConnections1):
	#TODO: verify calculate source to target connections only
	numberOfConnections2 = calculateSubgraphNumberIdenticalConcepts2(sentenceTreeNodeList, node1, nodeToCompare, 0)
	numberOfConnections1 += numberOfConnections2
	for subgraphNode1 in node1.graphNodeSourceList:	
		if(subgraphNode1 in sentenceTreeNodeList):	#verify subgraph instance was referenced in current sentence
			numberOfConnections1 = calculateSubgraphNumberIdenticalConcepts1(sentenceTreeNodeList, subgraphNode1, nodeToCompare, numberOfConnections1)
	return numberOfConnections1	

def calculateConceptRecencyLeafNode(sentenceTreeNodeList, lemma, currentTime):
	foundMostRecentInstanceNode, mostRecentInstanceNode, mostRecentInstanceTimeDiff = findMostRecentInstance(sentenceTreeNodeList, lemma, currentTime)
	if(not mostRecentInstanceNode):
		mostRecentInstanceTimeDiff = maxTimeDiff
	return mostRecentInstanceTimeDiff
	
def findMostRecentInstance(sentenceTreeNodeList, lemma, currentTime):	
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
			if(node2.activationTime != currentTime):	#ignore instances that were added from same sentence	#OR: node2 is not in(sentenceTreeNodeList)
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




#def identifyNodeType(node):


#reference similarity:
def findMostSimilarReferenceInGraph(sentenceTreeNodeList, node1, currentTime):
	foundReference = False
	referenceNode = None
	maxSimilarity = 0

	if(node1.lemma in graphNodeDictionary):
		#print("node1.lemma = ", node1.lemma)
		node1ConceptInstances = graphNodeDictionary[node1.lemma]	#current limitation: only reference identical lemmas [future allow referencing based on word vector similarity]
		for instanceID1, instanceNode1 in node1ConceptInstances.items():
			if(instanceNode1.activationTime != currentTime):	#ignore instances that were added from same sentence	#OR: instanceNode1 is not in(sentenceTreeNodeList)
				similarity = compareNodeReferenceSimilarity(sentenceTreeNodeList, node1, instanceNode1)
				recency = calculateRecencyAbsolute(instanceNode1.activationTime, currentTime)
				metric = calculateMetricReference(similarity, recency)
				if(metric > maxSimilarity):
					maxSimilarity = metric
					referenceNode = instanceNode1
					foundReference = True

	return foundReference, referenceNode, maxSimilarity
	
def compareNodeReferenceSimilarity(sentenceTreeNodeList, node1, node2):
	if(calculateReferenceSimilarityUsingWordVectorSimilarity):
		similarity = compareNodeWordVectorSimilarity(sentenceTreeNodeList, node1, node2)
	else:
		similarity = compareNodeIdenticalConceptSimilarity(sentenceTreeNodeList, node1, node2)
	#print("compareNodeReferenceSimilarity similarity = ", similarity)
	return similarity

def compareNodeIdenticalConceptSimilarity(sentenceTreeNodeList, node1, node2):
	if(calculateReferenceSimilarityBasedOnNodeSentenceSubgraphsDynamic):
		similarity = calculateSubgraphNumberIdenticalConcepts1(sentenceTreeNodeList, node1, node2, 0)			
	else:
		print("compareNodeIdenticalConceptSimilarity currently requires calculateReferenceSimilarityBasedOnNodeSentenceSubgraphsDynamic - no alternate method coded")
		exit()
		#similarity = calculateNumberIdenticalConcepts(node1, node2)
	return similarity

#compares all nodes in node1 subgraph (to nodeToCompare subgraphs)
#recurse node1 subgraph
def calculateSubgraphNumberIdenticalConcepts1(sentenceTreeNodeList, node1, nodeToCompare, numberOfConnections1):
	#TODO: verify calculate source to target connections only
	numberOfConnections2 = calculateSubgraphNumberIdenticalConcepts2(sentenceTreeNodeList, node1, nodeToCompare, 0)
	numberOfConnections1 += numberOfConnections2
	for subgraphNode1 in node1.graphNodeSourceList:	
		if(subgraphNode1 in sentenceTreeNodeList):	#verify subgraph instance was referenced in current sentence
			numberOfConnections1 = calculateSubgraphNumberIdenticalConcepts1(sentenceTreeNodeList, subgraphNode1, nodeToCompare, numberOfConnections1)
	return numberOfConnections1

#compares node1 with all nodes in node2 subgraph
#recurse node2 subgraph
def calculateSubgraphNumberIdenticalConcepts2(sentenceTreeNodeList, node1, node2, numberOfConnections):
	#TODO: verify calculate source to target connections only
	if(node1.lemma == node2.lemma):
		numberOfConnections += 1
	for subgraphNode2 in node2.graphNodeSourceList:	
		if(subgraphNode2 not in sentenceTreeNodeList):	#verify subgraph instance was not referenced in current sentence
			numberOfConnections = calculateSubgraphNumberIdenticalConcepts2(sentenceTreeNodeList, node1, subgraphNode2, numberOfConnections)
	return numberOfConnections
		
#def calculateNumberIdenticalConcepts(node, nodeEnd):
#	#CHECKTHIS: verify calculate source to target connections only
#	numberOfConnections = 0
#	instanceDict1 = graphNodeDictionary[node.lemma]
#	for instanceID1, node1 in instanceDict1.items():
#		for node2 in node1.graphNodeTargetList:
#			if(node1.lemma == nodeEnd.lemma):
#				numberOfConnections += 1
#	return numberOfConnections
	

#connection similarity:
def calculateFrequencyConnection(sentenceTreeNodeList, node1, node2):
	#CHECKTHIS: requires calibration
	#CHECKTHIS; note compares node subgraph source components (not target components)
	frequency = compareNodeConnectionSimilarity(sentenceTreeNodeList, node1, node2)	 #CHECKTHIS requires update - currently uses rudimentary word vector similarity comparison
	return frequency

def compareNodeConnectionSimilarity(sentenceTreeNodeList, node1, node2):
	if(calculateConnectionFrequencyUsingWordVectorSimilarity):
		similarity = compareNodeWordVectorSimilarity(sentenceTreeNodeList, node1, node2)		#compareNodeCorpusAssociationFrequency
	else:
		print("compareNodeCorpusAssociationFrequency currently requires calculateConnectionFrequencyUsingWordVectorSimilarity - no alternate method coded")
		exit()
	return similarity

def compareNodeWordVectorSimilarity(sentenceTreeNodeList, node1, node2):
	wordVectorDiff = compareNodeWordVector(sentenceTreeNodeList, node1, node2)
	#print("compareNodeWordVectorSimilarity: node1.lemma = ", node1.lemma, ", node2.lemma = ", node2.lemma, ", wordVectorDiff = ", wordVectorDiff)
	similarity = calculateWordVectorSimilarity(wordVectorDiff)
	return similarity
	
def compareNodeWordVector(sentenceTreeNodeList, node1, node2):
	if(calculateConnectionFrequencyBasedOnNodeSentenceSubgraphsDynamic):
		subgraphArtificalWordVector1 = calculateSubgraphArtificialWordVector(sentenceTreeNodeList, node1)
		subgraphArtificalWordVector2 = calculateSubgraphArtificialWordVector(sentenceTreeNodeList, node2)
		wordVectorDiff = compareWordVectors(subgraphArtificalWordVector1, subgraphArtificalWordVector2)		
	else:
		wordVectorDiff = compareWordVectors(node1.wordVector, node2.wordVector)
	return wordVectorDiff

def calculateSubgraphArtificialWordVector(sentenceTreeNodeList, node):
	subgraphArtificalWordVector = np.zeros(shape=ANNtf2_loadDataset.wordVectorLibraryNumDimensions)
	subgraphArtificalWordVector, subgraphSize = calculateSubgraphArtificialWordVectorRecurse(sentenceTreeNodeList, node, subgraphArtificalWordVector, 0)	
	subgraphArtificalWordVector = np.divide(subgraphArtificalWordVector, float(subgraphSize))
	return subgraphArtificalWordVector

def calculateSubgraphArtificialWordVectorRecurse(sentenceTreeNodeList, node, subgraphArtificalWordVector, subgraphSize):
	#CHECKTHIS: requires update - currently uses rudimentary combined word vector similarity comparison
	if(node.graphNodeType == graphNodeTypeLeaf):
		subgraphArtificalWordVector = np.add(subgraphArtificalWordVector, node.wordVector)
		#print("subgraphArtificalWordVector = ", np.mean(np.abs(subgraphArtificalWordVector)))
		subgraphSize = subgraphSize + 1
	for subgraphNode in node.graphNodeSourceList:	
		if(subgraphNode in sentenceTreeNodeList):	#verify subgraph instance was referenced in current sentence
			subgraphArtificalWordVector, subgraphSize = calculateSubgraphArtificialWordVectorRecurse(sentenceTreeNodeList, subgraphNode, subgraphArtificalWordVector, subgraphSize)
	return subgraphArtificalWordVector, subgraphSize
	
def calculateWordVectorSimilarity(wordVectorDiff):
	similarity = 1.0 - wordVectorDiff
	#print("similarity = ", similarity)
	return similarity

def compareWordVectors(wordVector1, wordVector2):
	wordVectorDiff = np.mean(np.absolute(np.subtract(wordVector1, wordVector2)))
	#print("\twordVector1 = ", wordVector1)
	#print("\twordVector2 = ", wordVector2)
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
	if(token.lemma_ == '-PRON-'):
		lemma = token.text	#https://stackoverflow.com/questions/56966754/how-can-i-make-spacy-not-produce-the-pron-lemma
	return lemma
		
def getTokenWordVector(token):
	wordVector = token.vector	#cpu: type numpy
	return wordVector

def getTokenPOStag(token):
	#nlp in context prediction only (not certain)
	posTag = token.pos_
	return posTag
  
def mean(lst):
	return sum(lst) / len(lst)

def minMeanMaxList(lst):
	return (min(lst), mean(lst), max(lst))


