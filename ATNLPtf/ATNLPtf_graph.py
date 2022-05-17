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

#calculateFrequency method: calculates frequency of co-occurance of words/subsentences in corpus 
calculateFrequencyUsingWordVectorSimilarity = True	#else calculateFrequencyUsingNumberOfConnectionsBetweenLemmasInGraph	#CHECKTHIS requires update - currently uses rudimentary word vector similarity comparison
calculateFrequencyBasedOnNodeSentenceSubgraphs = True	#else calculateFrequencyBasedOnNodes	#more advanced method of node similarity comparison #compares similarity of subgraphs of sentence nodes (rather than similarity of just the nodes themselves)

headNodePosTags = ["CC", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]

graphNodeTypeUnknown = 0
graphNodeTypeNormal = 1
graphNodeTypeHead = 2

conceptID = 0	#special instance ID for concepts
maxTimeDiff = 999999
minRecency = 0.1	#CHECKTHIS (> 0: ensures recency for unencountered concepts is not zero - required for metric)
maxTimeDiffForMatchingInstance = 2	#time in sentence index diff	#CHECKTHIS
metricThresholdToCreateConnection = 1.0	#requires calibration

class GraphNode:
	def __init__(self, instanceID, word, lemma, wordVector, posTag, nodeGraphType, activationTime):
		self.instanceID = instanceID
		self.word = word
		self.lemma = lemma
		self.wordVector = wordVector	#numpy array
		self.posTag = posTag	#nlp in context prediction only (not certain)
		self.graphNodeType = nodeGraphType
		self.activationTime = activationTime	#used to calculate recency
		#self.activationLevel = 0.0	#used to calculate recency
		#self.frequency = None	#float
		#connections;
		self.graphNodeTargets = {}	#dict indexed by lemma, every entry is a dictionary of GraphNode instances indexed by instanceID 
		self.graphNodeSources = {}	#dict indexed by lemma, every entry is a dictionary of GraphNode instances indexed by instanceID
		#self.foundRecentIndex = False	#temporary var (indicates referencing a previously declared instance in the article)
		
graphNodeDictionary = {}	#dict indexed by lemma, every entry is a dictionary of GraphNode instances indexed by instanceID (first instance is special; reserved for concept)
graphConnectionsDictionary = {}	#dict indexed tuples (lemma1, instanceID1, lemma2, instanceID2), every entry is a tuple of GraphNode instances/concepts (instanceNode1, instanceNode2) [directionality: 1=source, 2=target]
	#this is used for visualisation/fast lookup purposes only - can trace node graphNodeTargets/graphNodeSources instead
	

def generateGraph(sentenceIndex, sentence):
	
	print("\n\ngenerateGraph: sentenceIndex = ", sentenceIndex, "; ", sentence)
	
	currentTime = calculateActivationTime(sentenceIndex)
	
	sentenceNodeList = []	#local/temporary list of sentence instance nodes
		
	tokenisedSentence = tokeniseSentence(sentence)
	sentenceLength = len(tokenisedSentence)
	
	foundInstanceReferenceList = [False]*sentenceLength
	foundExistingConceptList = [False]*sentenceLength
	mostRecentInstanceNodeList = [None]*sentenceLength
	
	#add graph nodes;
	for w, token in enumerate(tokenisedSentence):	

		word = getTokenWord(token)
		lemma = getTokenLemma(token)
		wordVector = getTokenWordVector(token)	#numpy word vector
		posTag = getTokenPOStag(token)
		activationTime = calculateActivationTime(sentenceIndex)
		nodeGraphType = getNodeGraphType(posTag)
		
		foundRecentIndex = False
		if lemma in graphNodeDictionary:
			#lookup most recent instance
			foundMostRecentInstanceNode, mostRecentInstanceNode, mostRecentInstanceTimeDiff = findMostRecentInstance(lemma, currentTime)
			if(foundMostRecentInstanceNode):
				mostRecentInstanceNodeList[w] = mostRecentInstanceNode
				foundExistingConceptList[w] = True
				if(mostRecentInstanceTimeDiff < maxTimeDiffForMatchingInstance):
					foundRecentIndex = True
					foundInstanceReferenceList[w] = True
		else:
			#add concept to dictionary;
			createSubDictionaryForConcept(graphNodeDictionary, lemma)
			instanceID = conceptID
			conceptNode = GraphNode(instanceID, word, lemma, wordVector, posTag, nodeGraphType, currentTime)
			addInstanceNodeToGraph(lemma, instanceID, conceptNode)

		#add instance to dictionary;
		if(foundRecentIndex):
			print("create reference to mostRecentInstanceNode; ", mostRecentInstanceNode.lemma, ": instanceID=", mostRecentInstanceNode.instanceID) 
			instanceNode = mostRecentInstanceNode
		else:
			instanceID = getNewInstanceID(lemma)
			instanceNode = GraphNode(instanceID, word, lemma, wordVector, posTag, nodeGraphType, currentTime)
			print("create new instanceNode; ", instanceNode.lemma, ": instanceID=", instanceNode.instanceID)
			addInstanceNodeToGraph(lemma, instanceID, instanceNode)

		sentenceNodeList.append(instanceNode)


	#add graph connections;
	foundConnection = [False]*sentenceLength	#found connection (direction: source to target)
	for distance in range(1, sentenceLength):	#search for proximal connections before distal connections
		for w in range(sentenceLength):	
			if(not foundConnection[w]):
				w2 = w + distance
				connectionDirection = True	
				if(w2 < sentenceLength):
					node1 = sentenceNodeList[w]
					node2 = sentenceNodeList[w2]

					proximity = calculateProximity(w, w2)
					frequency = calculateFrequency(sentenceNodeList, node1, node2)
					if(foundExistingConceptList[w2]):	#or foundInstanceReferenceList[w2] - more stringent constraint
						mostRecentInstanceTimeDiff = calculateTimeDiffAbsolute(mostRecentInstanceNodeList[w2].activationTime, currentTime)	#regenerate value
						recency = calculateRecency(mostRecentInstanceTimeDiff)	#CHECKTHIS
					else:
						recency = minRecency
					metric = calculateMetric(proximity, frequency, recency)
					if(metric > metricThresholdToCreateConnection):
						print("create connection; w w2 = ", w, " ", w2, ", node1.lemma node2.lemma = ", node1.lemma, " ", node2.lemma, ", metric = ", metric)
						connectionDirection = True	#CHECKTHIS: always assume left to right directionality
						foundConnection[w] = True
						createGraphConnectionWrapper(node1, node2, connectionDirection)

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
	
def calculateMetric(proximity, frequency, recency):
	#print("\tproximity = ", proximity)
	#print("\tfrequency = ", frequency)
	#print("\trecency = ", recency)
	metric = proximity*frequency*recency #CHECKTHIS - normalisation of factors is required
	#print("\tmetric = ", metric)
	return metric
	
def calculateActivationTime(sentenceIndex):
	activationTime = sentenceIndex
	return activationTime

def calculateProximity(w, w2):
	proximity = abs(w-w2)
	return proximity
	
def calculateFrequency(sentenceNodeList, node1, node2):
	frequency = compareNodeSimilarity(sentenceNodeList, node1, node2)	 #CHECKTHIS requires update - currently uses rudimentary word vector similarity comparison
	return frequency

#FUTURE: needs to support subgraphFindMostRecentInstance (verify that subgraphs align)
def findMostRecentInstance(lemma, currentTime):
	#print("findMostRecentInstance") 
	#calculates recency of most recent instance, and returns this instance
	foundMostRecentInstanceNode = False
	instanceDict2 = graphNodeDictionary[lemma]
	minTimeDiff = maxTimeDiff
	mostRecentInstanceNode = None
	for instanceID2, node2 in instanceDict2.items():
		if(node2.activationTime != currentTime):	#ignore instances that were added from same sentence
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
	
def calculateRecency(timeDiff):
	recency = 1.0/timeDiff
	return recency
	
def calculateTimeDiffAbsolute(node2activationTime, currentTime):
	timeDiff = currentTime - node2activationTime
	return timeDiff

def createGraphConnectionWrapper(node1, node2, connectionDirection):
	if(connectionDirection):
		createGraphConnection(node1, node2)
	else:
		createGraphConnection(node2, node1)
		
def createGraphConnection(nodeSource, nodeTarget):
	addInstanceNodeToDictionary(nodeSource.graphNodeTargets, nodeTarget.lemma, nodeTarget.instanceID, nodeTarget)
	addInstanceNodeToDictionary(nodeTarget.graphNodeSources, nodeSource.lemma, nodeSource.instanceID, nodeSource)

	graphConnectionKey = createGraphConnectionKey(nodeSource, nodeTarget)
	graphConnectionsDictionary[graphConnectionKey] = (nodeSource, nodeTarget)

def createGraphConnectionKey(nodeSource, nodeTarget):
	connectionKey = (nodeSource.lemma, nodeSource.instanceID, nodeTarget.lemma, nodeTarget.instanceID)
	return connectionKey

#def identifyNodeType(node):


def compareNodeSimilarity(sentenceNodeList, node1, node2):
	if(calculateFrequencyBasedOnNodeSentenceSubgraphs):
		if(calculateFrequencyUsingWordVectorSimilarity):
			subgraphArtificalWordVector1 = node1.wordVector
			subgraphArtificalWordVector2 = node2.wordVector
			subgraphSize1 = calculateSubgraphArtificialWordVector(sentenceNodeList, node1, subgraphArtificalWordVector1, 1)
			subgraphSize2 = calculateSubgraphArtificialWordVector(sentenceNodeList, node2, subgraphArtificalWordVector2, 1)
			subgraphArtificalWordVector1 = np.divide(subgraphArtificalWordVector1, float(subgraphSize1))
			subgraphArtificalWordVector2 = np.divide(subgraphArtificalWordVector2, float(subgraphSize2))
			similarity = calculateWordVectorSimilarity(subgraphArtificalWordVector1, subgraphArtificalWordVector2)
		else:
			similarity = calculateSubgraphNumberConnections1(sentenceNodeList, node1, node2, 0)			
	else:
		if(calculateFrequencyUsingWordVectorSimilarity):
			similarity = calculateWordVectorSimilarity(node1.wordVector, node2.wordVector)
		else:
			similarity = calculateNumberConnections(node1, node2)
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


#compares all nodes in node1 subgraph (to nodeToCompare subgraphs)
#recurse node1 subgraph
def calculateSubgraphNumberConnections1(sentenceNodeList, node1, nodeToCompare, numberOfConnections1):
	#TODO: verify calculate source to target connections only
	numberOfConnections2 = calculateSubgraphNumberConnections2(sentenceNodeList, node1, nodeToCompare, 0)
	numberOfConnections1 += numberOfConnections2
	for subgraphInstanceID1, subgraphNode1 in node.graphNodeSources.items():	
		if(subgraphNode1 in sentenceNodeList):	#verify subgraph instance was referenced in current sentence
			numberOfConnections1 = calculateSubgraphNumberConnections1(sentenceNodeList, subgraphNode1, nodeToCompare, numberOfConnections1)
	return numberOfConnections1

#compares node1 with all nodes in node2 subgraph
#recurse node2 subgraph
def calculateSubgraphNumberConnections2(sentenceNodeList, node1, node2, numberOfConnections):
	#TODO: verify calculate source to target connections only
	if(node1.lemma == node2.lemma):
		numberOfConnections += 1
	for subgraphInstanceID2, subgraphNode2 in node2.graphNodeSources.items():	
		if(subgraphNode2 in sentenceNodeList):	#verify subgraph instance was referenced in current sentence
			numberOfConnections = calculateSubgraphNumberConnections2(sentenceNodeList, node1, subgraphNode2, numberOfConnections)
	return numberOfConnections
		
def calculateNumberConnections(node, nodeEnd):
	#TODO: verify calculate source to target connections only
	numberOfConnections = 0
	instanceDict1 = graphNodeDictionary[node.lemma]
	for instanceID1, node1 in instanceDict1.items():
		for instanceID2, node2 in node1.graphNodeTargets.items():
			if(node2.lemma == nodeEnd.lemma):
				numberOfConnections += 1
	return numberOfConnections
	
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
	
		
