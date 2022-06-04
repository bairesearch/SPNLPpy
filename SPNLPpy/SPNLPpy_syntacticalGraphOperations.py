"""SPNLPpy_syntacticalGraphOperations.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SPNLPpy_main.py

# Usage:
see SPNLPpy_main.py

# Description:
SPNLP Syntactical Graph Operations - common operations

"""

import numpy as np
import ANNtf2_loadDataset
from SPNLPpy_syntacticalNodeClass import *

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
			
def findInstanceNodeInGraph(syntacticalGraphNodeDictionary, lemma, instanceID):
	node = syntacticalGraphNodeDictionary[lemma][instanceID]
	return node

def addInstanceNodeToGraph(syntacticalGraphNodeDictionary, lemma, instanceID, node):
	addInstanceNodeToDictionary(syntacticalGraphNodeDictionary, lemma, instanceID, node)

def addInstanceNodeToDictionary(dic, lemma, instanceID, node):
	if lemma not in dic:
		createSubDictionaryForConcept(dic, lemma)
	dic[lemma][instanceID] = node

def isInstanceNodeInGraph(syntacticalGraphNodeDictionary, node):
	return isInstanceNodeInDictionary(syntacticalGraphNodeDictionary, node)
	
def isInstanceNodeInDictionary(dic, node):
	if node.lemma not in dic:
		createSubDictionaryForConcept(dic, node.lemma)
	if(node in dic[node.lemma]):
		result = True
	else:
		result = False
	return result
	
def getNewInstanceID(syntacticalGraphNodeDictionary, lemma):
	if lemma in syntacticalGraphNodeDictionary:
		#newInstanceID = len(syntacticalGraphNodeDictionary[lemma])	#not reliable in the event a sentence node temporarily added to dictionary was referenced and removed from dictionary
		if(len(syntacticalGraphNodeDictionary[lemma]) > 0):
			lastInstanceIDadded = list(syntacticalGraphNodeDictionary[lemma])[-1]	#get last element inserted in dictionary (will have highest instanceID) - requires Python 3.7
			newInstanceID = lastInstanceIDadded+1
		else:
			newInstanceID = 0
	else:
		newInstanceID = 0
	return newInstanceID

def removeNodeFromGraph(syntacticalGraphNodeDictionary, node):
	nodeConceptInstances = syntacticalGraphNodeDictionary[node.lemma]
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
	node1.CPsourceNodePosition = sourceNodePositionFirst
	node2.CPsourceNodePosition = sourceNodePositionSecond

	#if(addToConnectionsDictionary):
	#	graphConnectionKey = createGraphConnectionKey(hiddenNode, node1, node2)
	#	syntacticalGraphConnectionsDictionary[graphConnectionKey] = (hiddenNode, node1, node2)

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
def calculateFrequencyReference(CPsentenceTreeNodeList, node1, node2):
	#CHECKTHIS: requires calibration
	#CHECKTHIS; note compares node subgraph source components (not target components)
	frequency = compareNodeReferenceSimilarity(CPsentenceTreeNodeList, node1, node2)	 #CHECKTHIS requires update - currently uses rudimentary word vector similarity comparison
	return frequency

def compareNodeReferenceSimilarity(CPsentenceTreeNodeList, node1, node2):
	if(calculateReferenceFrequencyUsingWordVectorSimilarity):
		similarity = compareNodeWordVectorsSimilarity(CPsentenceTreeNodeList, node1, node2, calculateReferenceFrequencyBasedOnNodeSentenceSubgraphsDynamic)		#compareNodeCorpusAssociationFrequency
	else:
		similarity = compareNodeIdenticalConceptSimilarity(CPsentenceTreeNodeList, node1, node2, calculateReferenceFrequencyBasedOnNodeSentenceSubgraphsDynamic)
	return similarity
	
#frequency connection:
def calculateFrequencyConnection(CPsentenceTreeNodeList, node1, node2):
	#CHECKTHIS: requires calibration
	#CHECKTHIS; note compares node subgraph source components (not target components)
	frequency = compareNodeConnectionSimilarity(CPsentenceTreeNodeList, node1, node2)	 #CHECKTHIS requires update - currently uses rudimentary word vector similarity comparison
	return frequency

def compareNodeConnectionSimilarity(CPsentenceTreeNodeList, node1, node2):
	if(calculateConnectionFrequencyUsingWordVectorSimilarity):
		similarity = compareNodeWordVectorsSimilarity(CPsentenceTreeNodeList, node1, node2, calculateConnectionFrequencyBasedOnNodeSentenceSubgraphsDynamic)		#compareNodeCorpusAssociationFrequency
	else:
		print("compareNodeConnectionSimilarity currently requires calculateConnectionFrequencyUsingWordVectorSimilarity - no alternate method coded")
		exit()
	return similarity

#frequency metric 1 (word vector similarity):
def compareNodeWordVectorsSimilarity(CPsentenceTreeNodeList, node1, node2, nodeSentenceSubgraphsDynamic):
	wordVectorDiff = compareNodeWordVectors(CPsentenceTreeNodeList, node1, node2, nodeSentenceSubgraphsDynamic)
	#print("compareNodeWordVectorsSimilarity: node1.lemma = ", node1.lemma, ", node2.lemma = ", node2.lemma, ", wordVectorDiff = ", wordVectorDiff)
	similarity = calculateWordVectorSimilarity(wordVectorDiff)
	return similarity
	
def compareNodeWordVectors(CPsentenceTreeNodeList, node1, node2, nodeSentenceSubgraphsDynamic):
	if(nodeSentenceSubgraphsDynamic):
		subgraphArtificalWordVector1 = calculateSubgraphArtificialWordVector(CPsentenceTreeNodeList, node1)
		subgraphArtificalWordVector2 = calculateSubgraphArtificialWordVector(CPsentenceTreeNodeList, node2)
	else:
		subgraphArtificalWordVector1 = getBranchWordVector(node1)
		subgraphArtificalWordVector2 = getBranchWordVector(node2)
	wordVectorDiff = compareWordVectors(subgraphArtificalWordVector1, subgraphArtificalWordVector2)		
	return wordVectorDiff
	
def calculateSubgraphArtificialWordVector(CPsentenceTreeNodeList, node):
	subgraphArtificalWordVector = np.zeros(shape=ANNtf2_loadDataset.wordVectorLibraryNumDimensions)
	subgraphArtificalWordVector, CPsubgraphSize = calculateSubgraphArtificialWordVectorRecurse(CPsentenceTreeNodeList, node, subgraphArtificalWordVector, 0)	
	subgraphArtificalWordVector = np.divide(subgraphArtificalWordVector, float(CPsubgraphSize))
	return subgraphArtificalWordVector

def calculateSubgraphArtificialWordVectorRecurse(CPsentenceTreeNodeList, node, subgraphArtificalWordVector, CPsubgraphSize):
	#CHECKTHIS: requires update - currently uses rudimentary combined word vector similarity comparison
	if(node.graphNodeType == graphNodeTypeLeaf):
		subgraphArtificalWordVector = np.add(subgraphArtificalWordVector, node.wordVector)
		#print("subgraphArtificalWordVector = ", np.mean(np.abs(subgraphArtificalWordVector)))
		CPsubgraphSize = CPsubgraphSize + 1
	for subgraphNode in node.CPgraphNodeSourceList:	
		#if(subgraphNode in CPsentenceTreeNodeList):	#verify subgraph instance was referenced in current sentence
		subgraphArtificalWordVector, CPsubgraphSize = calculateSubgraphArtificialWordVectorRecurse(CPsentenceTreeNodeList, subgraphNode, subgraphArtificalWordVector, CPsubgraphSize)
	return subgraphArtificalWordVector, CPsubgraphSize
	
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
		wordVector = np.divide(np.add(connectionNode1.conceptWordVector, connectionNode2.conceptWordVector), (connectionNode1.CPsubgraphSize + connectionNode2.CPsubgraphSize))
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
		wordVector = np.divide(node1.conceptWordVector, node1.CPsubgraphSize)
	else:
		wordVector = np.divide(node1.wordVector)
	return wordVector

#frequency metric 2 (identical concept similarity):
def compareNodeIdenticalConceptSimilarity(CPsentenceTreeNodeList, node1, node2, nodeSentenceSubgraphsDynamic):
	if(nodeSentenceSubgraphsDynamic):
		similarity = calculateSubgraphNumberIdenticalConcepts(CPsentenceTreeNodeList, node1, node2)			
	else:
		print("compareNodeIdenticalConceptSimilarity currently requires nodeSentenceSubgraphsDynamic - no alternate method coded")
		exit()
	return similarity

def calculateSubgraphNumberIdenticalConcepts(CPsentenceTreeNodeList, node1, nodeToCompare):
	numberOfIdenticalConcepts, CPsubgraphSize = calculateSubgraphNumberIdenticalConcepts1(CPsentenceTreeNodeList, node1, nodeToCompare, 0, 0)
	similarity = numberOfIdenticalConcepts/CPsubgraphSize
	return similarity
	
#compares all nodes in node1 subgraph (to nodeToCompare subgraphs)
#recurse node1 subgraph
def calculateSubgraphNumberIdenticalConcepts1(CPsentenceTreeNodeList, node1, nodeToCompare, numberOfIdenticalConcepts1, CPsubgraphSize):
	#TODO: verify calculate source to target connections only
	numberOfIdenticalConcepts2 = calculateSubgraphNumberIdenticalConcepts2(CPsentenceTreeNodeList, node1, nodeToCompare, 0)
	numberOfIdenticalConcepts1 += numberOfIdenticalConcepts2
	CPsubgraphSize = CPsubgraphSize + 1
	for subgraphNode1 in node1.CPgraphNodeSourceList:	
		if(subgraphNode1 in CPsentenceTreeNodeList):	#verify subgraph instance was referenced in current sentence
			numberOfIdenticalConcepts1, CPsubgraphSize = calculateSubgraphNumberIdenticalConcepts1(CPsentenceTreeNodeList, subgraphNode1, nodeToCompare, numberOfIdenticalConcepts1, CPsubgraphSize)
	return numberOfIdenticalConcepts1, CPsubgraphSize

#compares node1 with all nodes in node2 subgraph
#recurse node2 subgraph
def calculateSubgraphNumberIdenticalConcepts2(CPsentenceTreeNodeList, node1, node2, numberOfIdenticalConcepts):
	#TODO: verify calculate source to target connections only
	if(node1.lemma == node2.lemma):
		numberOfIdenticalConcepts += 1
	for subgraphNode2 in node2.CPgraphNodeSourceList:	
		if(subgraphNode2 not in CPsentenceTreeNodeList):	#verify subgraph instance was not referenced in current sentence
			numberOfIdenticalConcepts = calculateSubgraphNumberIdenticalConcepts2(CPsentenceTreeNodeList, node1, subgraphNode2, numberOfIdenticalConcepts)
	return numberOfIdenticalConcepts



#recency:

#recency reference:	
def calculateRecencyReference(CPsentenceTreeNodeList, node1, node2, currentTime):
	#calculates recency based on concept last access time - no alternate method coded
	recency = compareNodeConceptTimeSimilarity(CPsentenceTreeNodeList, node1, node2, currentTime, calculateReferenceRecencyBasedOnNodeSentenceSubgraphsDynamic)
	return recency
	
#recency connection:
def calculateRecencyConnection(CPsentenceTreeNodeList, node1, node2, currentTime):
	#calculates recency based on concept last access time - no alternate method coded
	recency = compareNodeConceptTimeSimilarity(CPsentenceTreeNodeList, node1, node2, currentTime, calculateConnectionRecencyBasedOnNodeSentenceSubgraphsDynamic)
	return recency
	
#recency metric:	
def compareNodeConceptTimeSimilarity(CPsentenceTreeNodeList, node1, node2, currentTime, nodeSentenceSubgraphsDynamic):
	timeDiff = compareNodeConceptTime(CPsentenceTreeNodeList, node1, node2, currentTime, nodeSentenceSubgraphsDynamic)
	recencySimilarity = calculateRecency(timeDiff)
	return recencySimilarity
	
def compareNodeConceptTime(CPsentenceTreeNodeList, node1, node2, currentTime, nodeSentenceSubgraphsDynamic):
	#CHECKTHIS: requires calibration
	#CHECKTHIS: requires update - currently uses rudimentary combined minTimeDiff similarity comparison
	if(nodeSentenceSubgraphsDynamic):
		subgraphArtificalTime1 = calculateSubgraphArtificialTime(CPsentenceTreeNodeList, node1)
		subgraphArtificalTime2 = calculateSubgraphArtificialTime(CPsentenceTreeNodeList, node2)
	else:
		subgraphArtificalTime1 = getBranchConceptTime(node1)
		subgraphArtificalTime2 = getBranchConceptTime(node2)
	timeDiff = compareTime(subgraphArtificalTime1, subgraphArtificalTime2)
	return timeDiff
	
def calculateSubgraphArtificialTime(CPsentenceTreeNodeList, node):
	subgraphArtificalTime = 0
	subgraphArtificalTime, CPsubgraphSize = calculateSubgraphArtificialTimeRecurse(CPsentenceTreeNodeList, node, subgraphArtificalTime, 0)
	subgraphArtificalTime = (subgraphArtificalTime / CPsubgraphSize)
	return subgraphArtificalTime

def calculateSubgraphArtificialTimeRecurse(CPsentenceTreeNodeList, node, subgraphArtificalTime, CPsubgraphSize):
	if(node.graphNodeType == graphNodeTypeLeaf):
		subgraphArtificalTime = subgraphArtificalTime + node.conceptTime
		CPsubgraphSize = CPsubgraphSize + 1
	for subgraphNode in node.CPgraphNodeSourceList:	
		#if(subgraphNode in CPsentenceTreeNodeList):	#verify subgraph instance was referenced in current sentence
		subgraphArtificalTime, CPsubgraphSize = calculateSubgraphArtificialTimeRecurse(CPsentenceTreeNodeList, subgraphNode, subgraphArtificalTime, CPsubgraphSize)
	return subgraphArtificalTime, CPsubgraphSize

def getBranchConceptTime(node1):
	if(calculateRecencyBasedOnNodeSentenceSubgraphsDynamicEmulate):
		conceptTime = node1.conceptTime/node1.CPsubgraphSize
	else:
		conceptTime = node1.activationTime	#conceptTime
	return conceptTime

def compareTime(time1, time2):
	timeDiff = abs(time1 - time2)
	return timeDiff
	
#def calculateSubgraphMostRecentIdenticalConnection(CPsentenceTreeNodeList, node1, nodeToCompare, numberOfConnections1):
#	#TODO: verify calculate source to target connections only
#	numberOfConnections2 = calculateSubgraphNumberIdenticalConcepts2(CPsentenceTreeNodeList, node1, nodeToCompare, 0)
#	numberOfConnections1 += numberOfConnections2
#	for subgraphNode1 in node1.CPgraphNodeSourceList:	
#		#if(subgraphNode1 in CPsentenceTreeNodeList):	#verify subgraph instance was referenced in current sentence
#		numberOfConnections1 = calculateSubgraphNumberIdenticalConcepts1(CPsentenceTreeNodeList, subgraphNode1, nodeToCompare, numberOfConnections1)
#	return numberOfConnections1	

def calculateConceptTimeLeafNode(syntacticalGraphNodeDictionary, CPsentenceTreeNodeList, lemma, currentTime):
	foundMostRecentInstanceNode, mostRecentInstanceNode, mostRecentInstanceTimeDiff = findMostRecentInstance(syntacticalGraphNodeDictionary, CPsentenceTreeNodeList, lemma, currentTime)
	if(not mostRecentInstanceNode):
		mostRecentInstanceTimeDiff = maxTimeDiff
	return mostRecentInstanceTimeDiff
	
def findMostRecentInstance(syntacticalGraphNodeDictionary, CPsentenceTreeNodeList, lemma, currentTime):	
	#print("findMostRecentInstance") 
	#calculates recency of most recent instance, and returns this instance
	foundMostRecentInstanceNode = False
	mostRecentInstanceNode = None
	mostRecentInstanceTimeDiff = None
	if(lemma in syntacticalGraphNodeDictionary):
		instanceDict2 = syntacticalGraphNodeDictionary[lemma]
		minTimeDiff = maxTimeDiff
		mostRecentInstanceNode = None
		for instanceID2, node2 in instanceDict2.items():
			if(node2.activationTime != currentTime):	#ignore instances that were added from same sentence	#OR: node2 is not in(CPsentenceTreeNodeList)
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

def identifyBranchReferences(syntacticalGraphNodeDictionary, CPsentenceTreeNodeList, branchHeadNode, currentTime):
	for subgraphNode1 in branchHeadNode.CPgraphNodeSourceList:	
		#if(subgraphNode1 in CPsentenceTreeNodeList):	#verify subgraph instance was referenced in current sentence (should always be true)
		branchReferenceFound, branchReference, _ = identifyBranchReference(syntacticalGraphNodeDictionary, CPsentenceTreeNodeList, subgraphNode1, currentTime)
		if(branchReferenceFound):
			replaceBranch(syntacticalGraphNodeDictionary, branchHeadNode, subgraphNode1, branchReference)
		else:
			identifyBranchReferences(syntacticalGraphNodeDictionary, CPsentenceTreeNodeList, subgraphNode1, currentTime)

def identifyBranchReference(syntacticalGraphNodeDictionary, CPsentenceTreeNodeList, subgraphNode1, currentTime):
	#CHECKTHIS: current optimisation/lookup limitations;
		#subgraphNode1.lemma must match referenceNode.lemma
		#subgraphNode1 branch structure must match referenceNode branch structure
	
	foundReference = False
	referenceNode = None
	maxReferenceMetric = 0

	if(subgraphNode1.lemma in syntacticalGraphNodeDictionary):
		#print("subgraphNode1.lemma = ", subgraphNode1.lemma)
		node1ConceptInstances = syntacticalGraphNodeDictionary[subgraphNode1.lemma]	#current limitation: only reference identical lemmas [future allow referencing based on word vector similarity]
		for instanceID1, instanceNode1 in node1ConceptInstances.items():
			#print("\tinstanceNode1.activationTime = ", instanceNode1.activationTime)
			if(instanceNode1.activationTime != currentTime):	#ignore instances that were added from same sentence	#OR: instanceNode1 is not in(CPsentenceTreeNodeList)
				frequency = calculateFrequencyReference(CPsentenceTreeNodeList, subgraphNode1, instanceNode1)
				recency = calculateRecencyReference(CPsentenceTreeNodeList, subgraphNode1, instanceNode1, currentTime)
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
def replaceBranch(syntacticalGraphNodeDictionary, branchHeadNode, subgraphNode1, branchReference):

	branchHeadNode.CPgraphNodeSourceList.remove(subgraphNode1)
	branchHeadNode.CPgraphNodeSourceList.append(branchReference)
	branchReference.CPgraphNodeTargetList.append(branchHeadNode)
		
	#subgraphNode1.CPgraphNodeTargetList.clear()	#not necessary	
	deleteBranch(syntacticalGraphNodeDictionary, subgraphNode1)

#delete local branch (without references)
def deleteBranch(syntacticalGraphNodeDictionary, branchHeadNode):
	for subgraphNode1 in branchHeadNode.CPgraphNodeSourceList:	
		#if(subgraphNode1 in CPsentenceTreeNodeList):	#verify subgraph instance was referenced in current sentence (should always be true)
		removeNodeFromGraph(syntacticalGraphNodeDictionary, subgraphNode1)
		deleteBranch(syntacticalGraphNodeDictionary, subgraphNode1)
		del subgraphNode1
	



#python mean:

def mean(lst):
	return sum(lst) / len(lst)

def minMeanMaxList(lst):
	return (min(lst), mean(lst), max(lst))


