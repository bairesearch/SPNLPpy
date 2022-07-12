"""SPNLPpy_syntacticalGraphDependencyParserWordVectorsAcyclic.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SPNLPpy_main.py

# Usage:
see SPNLPpy_main.py

# Description:
SPNLP Syntactical Graph Dependency Parser (DP) Word Vectors - generate dependency parse tree/graph using input vectors (based on word proximity, frequency, and recency heuristics) via construction of intermediary acyclic graph
 
Preconditions: assumes leaf nodes already generated

"""

import numpy as np
import spacy
import ANNtf2_loadDataset
from SPNLPpy_syntacticalNodeClass import *
import SPNLPpy_syntacticalGraphOperations

printVerbose = False

calibrateConnectionMetricParameters = True

connectionNode1ToNode2 = True	#should not affect performance


def generateSyntacticalTreeDependencyParserWordVectorsAcyclic(sentenceIndex, sentenceLeafNodeList, sentenceTreeNodeList, connectivityStackNodeList, syntacticalGraphNodeDictionary):
		
	SPNLPpy_syntacticalGraphOperations.setParserType(syntacticalGraphTypeAcyclic)
	
	currentTime = SPNLPpy_syntacticalGraphOperations.calculateActivationTime(sentenceIndex)

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
					proximity = SPNLPpy_syntacticalGraphOperations.calculateProximityConnection(node1.w, node2.w)	#requires recalibration
					#proximity = 1.0
					frequency = SPNLPpy_syntacticalGraphOperations.calculateFrequencyConnection(sentenceTreeNodeList, node1, node2)
					recency = SPNLPpy_syntacticalGraphOperations.calculateRecencyConnection(sentenceTreeNodeList, node1, node2, currentTime)	#minimise the difference in concept last access recency between nodes
					connectionMetric = SPNLPpy_syntacticalGraphOperations.calculateMetricConnection(proximity, frequency, recency)
					if(printVerbose):
						print("calculateMetricConnection: node1.lemma = ", node1.lemma, ", node2.lemma = ", node2.lemma, ", connectionMetric = ", connectionMetric)			
					if(connectionMetric > SPNLPpy_syntacticalGraphOperations.metricThresholdToCreateConnection):
						if(connectionMetric > maxConnectionMetric):
							#if(not connectionExists(node1, node2)):	#redundant
							if(not traceGraphIsCyclicWrapper(node1, node2)):
								if(printVerbose):
									print("\tconnectionMetric found")
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
			print("generateSyntacticalTreeDependencyParserWordVectorsAcyclic: error connectionFound - check calculateMetricConnection parameters > 0.0, maxConnectionMetric = ", maxConnectionMetric)
			exit()

		if(printVerbose):
			print("create connection; w1 w2 = ", connectionNode1.w, " ", connectionNode2.w, ", connectionNode1.lemma connectionNode2.lemma = ", connectionNode1.lemma, " ", connectionNode2.lemma, ", metric = ", maxConnectionMetric)
		
		if(connectionNode1ToNode2):
			connectionNode1Temp = connectionNode1
			connectionNode1 = connectionNode2
			connectionNode2 = connectionNode1Temp
		
		#connection vars;
		SPNLPpy_syntacticalGraphOperations.createGraphConnectionAG(connectionNode1, connectionNode2)

		numberOfTracedNodesFound = traceGraphACcountSizeWrapper(connectionNode1)
		if(numberOfTracedNodesFound == len(sentenceLeafNodeList)):
			headNodeFound = True
			connectionNode1.graphNodeType = graphNodeTypeHead	#reference set delimiter (captures primary subject/action/object of sentence clause)
			graphHeadNode = connectionNode1
			
			createDependencyTreeFromAG(graphHeadNode, 0)
			
		
	if(calibrateConnectionMetricParameters):
		proximityMinMeanMax = SPNLPpy_syntacticalGraphOperations.minMeanMaxList(proximityList)
		frequencyMinMeanMax = SPNLPpy_syntacticalGraphOperations.minMeanMaxList(frequencyList)
		recencyMinMeanMax = SPNLPpy_syntacticalGraphOperations.minMeanMaxList(recencyList)
		metricMinMeanMax = SPNLPpy_syntacticalGraphOperations.minMeanMaxList(metricList)
		print("proximityMinMeanMax = ", proximityMinMeanMax)
		print("frequencyMinMeanMax = ", frequencyMinMeanMax)
		print("recencyMinMeanMax = ", recencyMinMeanMax)
		print("metricMinMeanMax = ", metricMinMeanMax)
			
	return graphHeadNode


def traceGraphACcountSizeWrapper(node1):
	numberOfTracedNodesFound = 0
	numberOfTracedNodesFound = traceGraphACcountSize(node1, numberOfTracedNodesFound)
	return numberOfTracedNodesFound

def traceGraphIsCyclicWrapper(node1, node2):
	return targetNodeExistsInSubgraph(node1, node2)	#connectionBetweenNodesExist

def targetNodeExistsInSubgraph(currentNode, targetNode):
	isCyclic = False
	if(currentNode == targetNode):
		isCyclic = True
	else:
		if(not currentNode.AGtraced):
			currentNode.AGtraced = True
			for connectionTarget in currentNode.AGconnectionList:
				if(targetNodeExistsInSubgraph(connectionTarget, targetNode)):
					isCyclic = True
			currentNode.AGtraced = False
	return isCyclic
			
def traceGraphACcountSize(currentNode, numberOfTracedNodesFound):
	if(not currentNode.AGtraced):
		currentNode.AGtraced = True
		numberOfTracedNodesFound += 1
		for connectionTarget in currentNode.AGconnectionList:
			numberOfTracedNodesFound = traceGraphACcountSize(connectionTarget, numberOfTracedNodesFound)
		currentNode.AGtraced = False
	return numberOfTracedNodesFound

def connectionExists(node1, node2):
	result = False
	if(node2 in node1.AGconnectionList):
		result = True
	return result	

def createDependencyTreeFromAG(currentNode, level):
	if(not currentNode.AGtraced):
		currentNode.AGtraced = True
		for connectionTarget in currentNode.AGconnectionList: 
			#print("connectionTarget = ", connectionTarget.word)
			if(not connectionTarget.AGtraced):
				SPNLPpy_syntacticalGraphOperations.createGraphConnectionDP(currentNode, connectionTarget)
			createDependencyTreeFromAG(connectionTarget, level+1)
		currentNode.DPtreeLevel = level
		currentNode.AGtraced = False
