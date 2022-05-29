"""ATNLPtf_syntacticalGraphOperations.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATNLPtf_main.py

# Usage:
see ATNLPtf_main.py

# Description:
ATNLP Syntactical Graph Operations - common operations

"""

import numpy as np
import ANNtf2_loadDataset
from ATNLPtf_syntacticalNodeClass import *

printVerbose = False

#calculateFrequencyConnection method: calculates frequency of co-occurance of words/subsentences in corpus 
calculateConnectionFrequencyUsingWordVectorSimilarity = True	#mandatory	#CHECKTHIS requires update - currently uses rudimentary word vector similarity comparison
calculateReferenceFrequencyUsingWordVectorSimilarity = True	#optional	#else calculate reference similarity using calculateSubgraphNumberIdenticalConcepts

calculateConnectionFrequencyBasedOnNodeSentenceSubgraphsDynamic = False	#optional (not required to compare hidden node similarity as artificial word vectors are generated for new hidden nodes in sentence)	#else getBranchWordVector (flat)
calculateConnectionRecencyBasedOnNodeSentenceSubgraphsDynamic = False	#optional (not required to compare hidden node concept recency as artificial concept recency values are generated for new hidden nodes in sentence)		#else getBranchConceptTime (flat)
if(calculateReferenceFrequencyUsingWordVectorSimilarity):
	calculateReferenceFrequencyBasedOnNodeSentenceSubgraphsDynamic = False	#mandatory
else:
	calculateReferenceFrequencyBasedOnNodeSentenceSubgraphsDynamic = True	#mandatory
calculateReferenceRecencyBasedOnNodeSentenceSubgraphsDynamic = False	#mandatory

calculateFrequencyBasedOnNodeSentenceSubgraphsDynamicEmulate = True	#optional - branch wordVector calculated based on average of leafNode wordVectors, else average of previous branch wordVectors
calculateRecencyBasedOnNodeSentenceSubgraphsDynamicEmulate = True	#optional - branch conceptTime calculated based on average of leafNode conceptTime, else average of previous branch conceptTimes

conceptID = 0	#special instance ID for concepts
maxTimeDiff = 10.0	#calculateTimeDiff(minRecency)	#CHECKTHIS: requires calibration (<= ~10: ensures timeDiff for unencountered concepts is not infinite - required for metric)	#units: sentenceIndex
#minRecency = calculateRecency(maxTimeDiff)	#  minRecency = 0.1	#CHECKTHIS: requires calibration (> 0: ensures recency for unencountered concepts is not zero - required for metric)	
maxRecency = 1.0
maxTimeDiffForMatchingInstance = 2	#time in sentence index diff	#CHECKTHIS: requires calibration
metricThresholdToCreateConnection = 0.0	#CHECKTHIS: requires calibration
metricThresholdToCreateReference = 1.0	#CHECKTHIS: requires calibration



#node:

def createSubDictionaryForConcept(dic, lemma):
	dic[lemma] = {}	#create empty dictionary for new concept
			
def findInstanceNodeInGraph(graphNodeDictionary, lemma, instanceID):
	node = graphNodeDictionary[lemma][instanceID]
	return node

def addInstanceNodeToGraph(graphNodeDictionary, lemma, instanceID, node):
	addInstanceNodeToDictionary(graphNodeDictionary, lemma, instanceID, node)

def addInstanceNodeToDictionary(dic, lemma, instanceID, node):
	if lemma not in dic:
		createSubDictionaryForConcept(dic, lemma)
	dic[lemma][instanceID] = node

def isInstanceNodeInGraph(graphNodeDictionary, node):
	return isInstanceNodeInDictionary(graphNodeDictionary, node)
	
def isInstanceNodeInDictionary(dic, node):
	if node.lemma not in dic:
		createSubDictionaryForConcept(dic, node.lemma)
	if(node in dic[node.lemma]):
		result = True
	else:
		result = False
	return result
	
def getNewInstanceID(graphNodeDictionary, lemma):
	if lemma in graphNodeDictionary:
		#newInstanceID = len(graphNodeDictionary[lemma])	#not reliable in the event a sentence node temporarily added to dictionary was referenced and removed from dictionary
		if(len(graphNodeDictionary[lemma]) > 0):
			lastInstanceIDadded = list(graphNodeDictionary[lemma])[-1]	#get last element inserted in dictionary (will have highest instanceID) - requires Python 3.7
			newInstanceID = lastInstanceIDadded+1
		else:
			newInstanceID = 0
	else:
		newInstanceID = 0
	return newInstanceID

def removeNodeFromGraph(graphNodeDictionary, node):
	nodeConceptInstances = graphNodeDictionary[node.lemma]
	nodeConceptInstances.pop(node.instanceID) 

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



#metric:

def calculateMetricReference(frequency, recency):
	#print("\tfrequency = ", frequency)
	#print("\trecency = ", recency)
	metric = frequency*recency #CHECKTHIS: requires calibration - normalisation of factors is required
	#print("\tmetric = ", metric)
	return metric
		
def calculateMetricConnection(proximity, frequency, recency):
	metric = proximity*frequency*recency #CHECKTHIS: requires calibration - normalisation of factors is required
	#print("\t\tcalculateMetricConnection: metric = ", metric, "; proximity = ", proximity, ", frequency = ", frequency, ", recency = ", recency)
	return metric
	
	
	
#proximity:

def calculateProximityConnection(w, w2):
	proximity = 1.0 / abs(w-w2)	#CHECKTHIS: requires calibration
	#proximity = 1.0	#complete deweight of proximity parameter
	return proximity



#frequency:

#frequency reference:
def calculateFrequencyReference(sentenceTreeNodeList, node1, node2):
	#CHECKTHIS: requires calibration
	#CHECKTHIS; note compares node subgraph source components (not target components)
	frequency = compareNodeReferenceSimilarity(sentenceTreeNodeList, node1, node2)	 #CHECKTHIS requires update - currently uses rudimentary word vector similarity comparison
	return frequency

def compareNodeReferenceSimilarity(sentenceTreeNodeList, node1, node2):
	if(calculateReferenceFrequencyUsingWordVectorSimilarity):
		similarity = compareNodeWordVectorsSimilarity(sentenceTreeNodeList, node1, node2, calculateReferenceFrequencyBasedOnNodeSentenceSubgraphsDynamic)		#compareNodeCorpusAssociationFrequency
	else:
		similarity = compareNodeIdenticalConceptSimilarity(sentenceTreeNodeList, node1, node2, calculateReferenceFrequencyBasedOnNodeSentenceSubgraphsDynamic)
	return similarity
	
#frequency connection:
def calculateFrequencyConnection(sentenceTreeNodeList, node1, node2):
	#CHECKTHIS: requires calibration
	#CHECKTHIS; note compares node subgraph source components (not target components)
	frequency = compareNodeConnectionSimilarity(sentenceTreeNodeList, node1, node2)	 #CHECKTHIS requires update - currently uses rudimentary word vector similarity comparison
	return frequency

def compareNodeConnectionSimilarity(sentenceTreeNodeList, node1, node2):
	if(calculateConnectionFrequencyUsingWordVectorSimilarity):
		similarity = compareNodeWordVectorsSimilarity(sentenceTreeNodeList, node1, node2, calculateConnectionFrequencyBasedOnNodeSentenceSubgraphsDynamic)		#compareNodeCorpusAssociationFrequency
	else:
		print("compareNodeConnectionSimilarity currently requires calculateConnectionFrequencyUsingWordVectorSimilarity - no alternate method coded")
		exit()
	return similarity

#frequency metric 1 (word vector similarity):
def compareNodeWordVectorsSimilarity(sentenceTreeNodeList, node1, node2, nodeSentenceSubgraphsDynamic):
	wordVectorDiff = compareNodeWordVectors(sentenceTreeNodeList, node1, node2, nodeSentenceSubgraphsDynamic)
	#print("compareNodeWordVectorsSimilarity: node1.lemma = ", node1.lemma, ", node2.lemma = ", node2.lemma, ", wordVectorDiff = ", wordVectorDiff)
	similarity = calculateWordVectorSimilarity(wordVectorDiff)
	return similarity
	
def compareNodeWordVectors(sentenceTreeNodeList, node1, node2, nodeSentenceSubgraphsDynamic):
	if(nodeSentenceSubgraphsDynamic):
		subgraphArtificalWordVector1 = calculateSubgraphArtificialWordVector(sentenceTreeNodeList, node1)
		subgraphArtificalWordVector2 = calculateSubgraphArtificialWordVector(sentenceTreeNodeList, node2)
	else:
		subgraphArtificalWordVector1 = getBranchWordVector(node1)
		subgraphArtificalWordVector2 = getBranchWordVector(node2)
	wordVectorDiff = compareWordVectors(subgraphArtificalWordVector1, subgraphArtificalWordVector2)		
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
		#if(subgraphNode in sentenceTreeNodeList):	#verify subgraph instance was referenced in current sentence
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

def getBranchWordVectorFromSourceNodes(connectionNode1, connectionNode2):
	if(calculateFrequencyBasedOnNodeSentenceSubgraphsDynamicEmulate):
		wordVector = np.divide(np.add(connectionNode1.conceptWordVector, connectionNode2.conceptWordVector), (connectionNode1.subgraphSize + connectionNode2.subgraphSize))
	else:
		wordVector = np.divide(np.add(connectionNode1.wordVector, connectionNode2.wordVector), 2.0)
	return wordVector

def getBranchWordVectorFromSourceNodesSum(wordVectorConnectionNodeSum, conceptWordVectorConnectionNodeSum, subgraphSizeConnectionNodeSum, numberOfConnectionNodes):
	if(calculateFrequencyBasedOnNodeSentenceSubgraphsDynamicEmulate):
		wordVector = np.divide(conceptWordVectorConnectionNodeSum, subgraphSizeConnectionNodeSum)
	else:
		wordVector = np.divide(wordVectorConnectionNodeSum, numberOfConnectionNodes)
	return wordVector
	
	
def getBranchWordVector(node1):
	if(calculateFrequencyBasedOnNodeSentenceSubgraphsDynamicEmulate):
		wordVector = np.divide(node1.conceptWordVector, node1.subgraphSize)
	else:
		wordVector = np.divide(node1.wordVector)
	return wordVector

#frequency metric 2 (identical concept similarity):
def compareNodeIdenticalConceptSimilarity(sentenceTreeNodeList, node1, node2, nodeSentenceSubgraphsDynamic):
	if(nodeSentenceSubgraphsDynamic):
		similarity = calculateSubgraphNumberIdenticalConcepts(sentenceTreeNodeList, node1, node2)			
	else:
		print("compareNodeIdenticalConceptSimilarity currently requires nodeSentenceSubgraphsDynamic - no alternate method coded")
		exit()
	return similarity

def calculateSubgraphNumberIdenticalConcepts(sentenceTreeNodeList, node1, nodeToCompare):
	numberOfIdenticalConcepts, subgraphSize = calculateSubgraphNumberIdenticalConcepts1(sentenceTreeNodeList, node1, nodeToCompare, 0, 0)
	similarity = numberOfIdenticalConcepts/subgraphSize
	return similarity
	
#compares all nodes in node1 subgraph (to nodeToCompare subgraphs)
#recurse node1 subgraph
def calculateSubgraphNumberIdenticalConcepts1(sentenceTreeNodeList, node1, nodeToCompare, numberOfIdenticalConcepts1, subgraphSize):
	#TODO: verify calculate source to target connections only
	numberOfIdenticalConcepts2 = calculateSubgraphNumberIdenticalConcepts2(sentenceTreeNodeList, node1, nodeToCompare, 0)
	numberOfIdenticalConcepts1 += numberOfIdenticalConcepts2
	subgraphSize = subgraphSize + 1
	for subgraphNode1 in node1.graphNodeSourceList:	
		if(subgraphNode1 in sentenceTreeNodeList):	#verify subgraph instance was referenced in current sentence
			numberOfIdenticalConcepts1, subgraphSize = calculateSubgraphNumberIdenticalConcepts1(sentenceTreeNodeList, subgraphNode1, nodeToCompare, numberOfIdenticalConcepts1, subgraphSize)
	return numberOfIdenticalConcepts1, subgraphSize

#compares node1 with all nodes in node2 subgraph
#recurse node2 subgraph
def calculateSubgraphNumberIdenticalConcepts2(sentenceTreeNodeList, node1, node2, numberOfIdenticalConcepts):
	#TODO: verify calculate source to target connections only
	if(node1.lemma == node2.lemma):
		numberOfIdenticalConcepts += 1
	for subgraphNode2 in node2.graphNodeSourceList:	
		if(subgraphNode2 not in sentenceTreeNodeList):	#verify subgraph instance was not referenced in current sentence
			numberOfIdenticalConcepts = calculateSubgraphNumberIdenticalConcepts2(sentenceTreeNodeList, node1, subgraphNode2, numberOfIdenticalConcepts)
	return numberOfIdenticalConcepts



#recency:

#recency reference:	
def calculateRecencyReference(sentenceTreeNodeList, node1, node2, currentTime):
	#calculates recency based on concept last access time - no alternate method coded
	recency = compareNodeConceptTimeSimilarity(sentenceTreeNodeList, node1, node2, currentTime, calculateReferenceRecencyBasedOnNodeSentenceSubgraphsDynamic)
	return recency
	
#recency connection:
def calculateRecencyConnection(sentenceTreeNodeList, node1, node2, currentTime):
	#calculates recency based on concept last access time - no alternate method coded
	recency = compareNodeConceptTimeSimilarity(sentenceTreeNodeList, node1, node2, currentTime, calculateConnectionRecencyBasedOnNodeSentenceSubgraphsDynamic)
	return recency
	
#recency metric:	
def compareNodeConceptTimeSimilarity(sentenceTreeNodeList, node1, node2, currentTime, nodeSentenceSubgraphsDynamic):
	timeDiff = compareNodeConceptTime(sentenceTreeNodeList, node1, node2, currentTime, nodeSentenceSubgraphsDynamic)
	recencySimilarity = calculateRecency(timeDiff)
	return recencySimilarity
	
def compareNodeConceptTime(sentenceTreeNodeList, node1, node2, currentTime, nodeSentenceSubgraphsDynamic):
	#CHECKTHIS: requires calibration
	#CHECKTHIS: requires update - currently uses rudimentary combined minTimeDiff similarity comparison
	if(nodeSentenceSubgraphsDynamic):
		subgraphArtificalTime1 = calculateSubgraphArtificialTime(sentenceTreeNodeList, node1)
		subgraphArtificalTime2 = calculateSubgraphArtificialTime(sentenceTreeNodeList, node2)
	else:
		subgraphArtificalTime1 = getBranchConceptTime(node1)
		subgraphArtificalTime2 = getBranchConceptTime(node2)
	timeDiff = compareTime(subgraphArtificalTime1, subgraphArtificalTime2)
	return timeDiff
	
def calculateSubgraphArtificialTime(sentenceTreeNodeList, node):
	subgraphArtificalTime = 0
	subgraphArtificalTime, subgraphSize = calculateSubgraphArtificialTimeRecurse(sentenceTreeNodeList, node, subgraphArtificalTime, 0)
	subgraphArtificalTime = (subgraphArtificalTime / subgraphSize)
	return subgraphArtificalTime

def calculateSubgraphArtificialTimeRecurse(sentenceTreeNodeList, node, subgraphArtificalTime, subgraphSize):
	if(node.graphNodeType == graphNodeTypeLeaf):
		subgraphArtificalTime = subgraphArtificalTime + node.conceptTime
		subgraphSize = subgraphSize + 1
	for subgraphNode in node.graphNodeSourceList:	
		#if(subgraphNode in sentenceTreeNodeList):	#verify subgraph instance was referenced in current sentence
		subgraphArtificalTime, subgraphSize = calculateSubgraphArtificialTimeRecurse(sentenceTreeNodeList, subgraphNode, subgraphArtificalTime, subgraphSize)
	return subgraphArtificalTime, subgraphSize

def getBranchConceptTime(node1):
	if(calculateRecencyBasedOnNodeSentenceSubgraphsDynamicEmulate):
		conceptTime = node1.conceptTime/node1.subgraphSize
	else:
		conceptTime = node1.activationTime	#conceptTime
	return conceptTime

def compareTime(time1, time2):
	timeDiff = abs(time1 - time2)
	return timeDiff
	
#def calculateSubgraphMostRecentIdenticalConnection(sentenceTreeNodeList, node1, nodeToCompare, numberOfConnections1):
#	#TODO: verify calculate source to target connections only
#	numberOfConnections2 = calculateSubgraphNumberIdenticalConcepts2(sentenceTreeNodeList, node1, nodeToCompare, 0)
#	numberOfConnections1 += numberOfConnections2
#	for subgraphNode1 in node1.graphNodeSourceList:	
#		#if(subgraphNode1 in sentenceTreeNodeList):	#verify subgraph instance was referenced in current sentence
#		numberOfConnections1 = calculateSubgraphNumberIdenticalConcepts1(sentenceTreeNodeList, subgraphNode1, nodeToCompare, numberOfConnections1)
#	return numberOfConnections1	

def calculateConceptTimeLeafNode(graphNodeDictionary, sentenceTreeNodeList, lemma, currentTime):
	foundMostRecentInstanceNode, mostRecentInstanceNode, mostRecentInstanceTimeDiff = findMostRecentInstance(graphNodeDictionary, sentenceTreeNodeList, lemma, currentTime)
	if(not mostRecentInstanceNode):
		mostRecentInstanceTimeDiff = maxTimeDiff
	return mostRecentInstanceTimeDiff
	
def findMostRecentInstance(graphNodeDictionary, sentenceTreeNodeList, lemma, currentTime):	
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

def calculateActivationTime(sentenceIndex):
	activationTime = sentenceIndex
	return activationTime
	
#not currently used;
def compareNodeLastActivationTime(node1, node2):
	timeDiffConnection = compareTime(node1.activationTime, node2.activationTime)
	return timeDiffConnection



#referencing:

def identifyBranchReferences(graphNodeDictionary, sentenceTreeNodeList, branchHeadNode, currentTime):
	for subgraphNode1 in branchHeadNode.graphNodeSourceList:	
		#if(subgraphNode1 in sentenceTreeNodeList):	#verify subgraph instance was referenced in current sentence (should always be true)
		branchReferenceFound, branchReference, _ = identifyBranchReference(graphNodeDictionary, sentenceTreeNodeList, subgraphNode1, currentTime)
		if(branchReferenceFound):
			replaceBranch(graphNodeDictionary, branchHeadNode, subgraphNode1, branchReference)
		else:
			identifyBranchReferences(graphNodeDictionary, sentenceTreeNodeList, subgraphNode1, currentTime)

def identifyBranchReference(graphNodeDictionary, sentenceTreeNodeList, subgraphNode1, currentTime):
	#CHECKTHIS: current optimisation/lookup limitations;
		#subgraphNode1.lemma must match referenceNode.lemma
		#subgraphNode1 branch structure must match referenceNode branch structure
	
	foundReference = False
	referenceNode = None
	maxReferenceMetric = 0

	if(subgraphNode1.lemma in graphNodeDictionary):
		#print("subgraphNode1.lemma = ", subgraphNode1.lemma)
		node1ConceptInstances = graphNodeDictionary[subgraphNode1.lemma]	#current limitation: only reference identical lemmas [future allow referencing based on word vector similarity]
		for instanceID1, instanceNode1 in node1ConceptInstances.items():
			#print("\tinstanceNode1.activationTime = ", instanceNode1.activationTime)
			if(instanceNode1.activationTime != currentTime):	#ignore instances that were added from same sentence	#OR: instanceNode1 is not in(sentenceTreeNodeList)
				frequency = calculateFrequencyReference(sentenceTreeNodeList, subgraphNode1, instanceNode1)
				recency = calculateRecencyReference(sentenceTreeNodeList, subgraphNode1, instanceNode1, currentTime)
				referenceMetric = calculateMetricReference(frequency, recency)
				#print("identifyBranchReference:")
				#print("\tfrequency = ", frequency)
				#print("\trecency = ", recency)
				#print("\treferenceMetric = ", referenceMetric)
				if(referenceMetric > metricThresholdToCreateReference):
					if(referenceMetric > maxReferenceMetric):
						print("identifyBranchReference: foundReference, referenceMetric = ", referenceMetric)
						print("\tfrequency = ", frequency)
						print("\trecency = ", recency)
						maxReferenceMetric = referenceMetric
						referenceNode = instanceNode1
						foundReference = True

	return foundReference, referenceNode, maxReferenceMetric
		
#replace local branch with referenced graph branch
def replaceBranch(graphNodeDictionary, branchHeadNode, subgraphNode1, branchReference):

	branchHeadNode.graphNodeSourceList.remove(subgraphNode1)
	branchHeadNode.graphNodeSourceList.append(branchReference)
	branchReference.graphNodeTargetList.append(branchHeadNode)
		
	#subgraphNode1.graphNodeTargetList.clear()	#not necessary	
	deleteBranch(graphNodeDictionary, subgraphNode1)

#delete local branch (without references)
def deleteBranch(graphNodeDictionary, branchHeadNode):
	for subgraphNode1 in branchHeadNode.graphNodeSourceList:	
		#if(subgraphNode1 in sentenceTreeNodeList):	#verify subgraph instance was referenced in current sentence (should always be true)
		removeNodeFromGraph(graphNodeDictionary, subgraphNode1)
		deleteBranch(graphNodeDictionary, subgraphNode1)
		del subgraphNode1
	



#python mean:

def mean(lst):
	return sum(lst) / len(lst)

def minMeanMaxList(lst):
	return (min(lst), mean(lst), max(lst))


