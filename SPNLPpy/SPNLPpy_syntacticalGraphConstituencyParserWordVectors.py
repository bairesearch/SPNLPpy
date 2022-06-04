"""SPNLPpy_syntacticalGraphConstituencyParserWordVectors.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SPNLPpy_main.py

# Usage:
see SPNLPpy_main.py

# Description:
SPNLP Syntactical Graph Constituency Parser (CP) Word Vectors - generate constituency parse tree/graph using input vectors (based on word proximity, frequency, and recency heuristics)

Preconditions: assumes leaf nodes already generated

"""

import numpy as np
import spacy
import ANNtf2_loadDataset
from SPNLPpy_syntacticalNodeClass import *
import SPNLPpy_syntacticalGraphOperations

calibrateConnectionMetricParameters = True

def generateSyntacticalTreeConstituencyParserWordVectors(sentenceIndex, sentenceLeafNodeList, CPsentenceTreeNodeList, CPconnectivityStackNodeList, syntacticalGraphNodeDictionary):

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

		for node1StackIndex, node1 in enumerate(CPconnectivityStackNodeList):
			for node2StackIndex, node2 in enumerate(CPconnectivityStackNodeList):
				if(node1StackIndex != node2StackIndex):
					#print("node1.CPwMax = ", node1.CPwMax)
					#print("node2.CPwMin = ", node2.CPwMin)
					if(node1.CPwMax+1 == node2.CPwMin):
						if(SPNLPpy_syntacticalGraphOperations.printVerbose):
							print("calculateMetricConnection: node1.lemma = ", node1.lemma, ", node2.lemma = ", node2.lemma)
						proximity = SPNLPpy_syntacticalGraphOperations.calculateProximityConnection(node1.w, node2.w)
						frequency = SPNLPpy_syntacticalGraphOperations.calculateFrequencyConnection(CPsentenceTreeNodeList, node1, node2)
						recency = SPNLPpy_syntacticalGraphOperations.calculateRecencyConnection(CPsentenceTreeNodeList, node1, node2, currentTime)	#minimise the difference in concept last access recency between left/right node
						connectionMetric = SPNLPpy_syntacticalGraphOperations.calculateMetricConnection(proximity, frequency, recency)
						if(connectionMetric > SPNLPpy_syntacticalGraphOperations.metricThresholdToCreateConnection):
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

		if(SPNLPpy_syntacticalGraphOperations.printVerbose):
			print("create connection; w1 w2 = ", connectionNode1.w, " ", connectionNode2.w, ", connectionNode1.lemma connectionNode2.lemma = ", connectionNode1.lemma, " ", connectionNode2.lemma, ", metric = ", maxConnectionMetric)

		#CHECKTHIS limitation - infers directionality (source/target) of connection based on w1/w2 word order		
		connectionDirection = True	#CHECKTHIS: always assume left to right directionality

		#primary vars;
		word = connectionNode1.word + connectionNode2.word
		lemma = connectionNode1.lemma + connectionNode2.lemma
		wordVector = SPNLPpy_syntacticalGraphOperations.getBranchWordVectorFromSourceNodes(connectionNode1, connectionNode2)
		posTag = None
		activationTime = SPNLPpy_syntacticalGraphOperations.calculateActivationTime(sentenceIndex)	#mean([connectionNode1.activationTime, connectionNode2.activationTime]) 
		nodeGraphType = graphNodeTypeBranch

		#sentenceTreeArtificial vars;
		CPsubgraphSize = connectionNode1.CPsubgraphSize + connectionNode2.CPsubgraphSize + 1
		conceptWordVector = np.add(connectionNode1.conceptWordVector, connectionNode2.conceptWordVector)
		conceptTime = connectionNode1.conceptTime + connectionNode2.conceptTime
		CPtreeLevel = max(connectionNode1.CPtreeLevel, connectionNode2.CPtreeLevel) + 1
		w = SPNLPpy_syntacticalGraphOperations.mean([connectionNode1.w, connectionNode2.w])
		CPwMin = min(connectionNode1.CPwMin, connectionNode2.CPwMin)
		CPwMax = max(connectionNode1.CPwMax, connectionNode2.CPwMax)

		instanceID = SPNLPpy_syntacticalGraphOperations.getNewInstanceID(syntacticalGraphNodeDictionary, lemma)
		hiddenNode = SyntacticalNode(instanceID, word, lemma, wordVector, posTag, nodeGraphType, currentTime, CPsubgraphSize, conceptWordVector, conceptTime, w, CPwMin, CPwMax, CPtreeLevel, sentenceIndex)
		SPNLPpy_syntacticalGraphOperations.addInstanceNodeToGraph(syntacticalGraphNodeDictionary, lemma, instanceID, hiddenNode)
		
		#connection vars;
		SPNLPpy_syntacticalGraphOperations.createGraphConnectionWrapper(hiddenNode, connectionNode1, connectionNode2, connectionDirection, addToConnectionsDictionary=False)
		CPsentenceTreeNodeList.append(hiddenNode)
		CPconnectivityStackNodeList.remove(connectionNode1)
		CPconnectivityStackNodeList.remove(connectionNode2)
		CPconnectivityStackNodeList.append(hiddenNode)

		if(len(CPconnectivityStackNodeList) == 1):
			headNodeFound = True
			hiddenNode.graphNodeType = graphNodeTypeHead	#reference set delimiter (captures primary subject/action/object of sentence clause)
			graphHeadNode = hiddenNode
			
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

