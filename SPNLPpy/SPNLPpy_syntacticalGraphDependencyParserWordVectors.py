"""SPNLPpy_syntacticalGraphDependencyParserWordVectors.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SPNLPpy_main.py

# Usage:
see SPNLPpy_main.py

# Description:
SPNLP Syntactical Graph Dependency Parser (DP) Word Vectors - generate dependency parse tree/graph using input vectors (based on word proximity, frequency, and recency heuristics)
 
Preconditions: assumes leaf nodes already generated

"""

import numpy as np
import spacy
import ANNtf2_loadDataset
from SPNLPpy_syntacticalNodeClass import *
import SPNLPpy_syntacticalGraphOperations

calibrateConnectionMetricParameters = True

interpretRightNodeAsGovernor = True
 
def generateSyntacticalTreeDependencyParserWordVectors(sentenceIndex, sentenceLeafNodeList, sentenceTreeNodeList, connectivityStackNodeList, syntacticalGraphNodeDictionary):

	useDependencyParseTree = True
	SPNLPpy_syntacticalGraphOperations.setParserType(useDependencyParseTree)
	
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
					#print("node1.DPwMax = ", node1.DPwMax)
					#print("node2.DPwMin = ", node2.DPwMin)
					if(node1.DPwMax+1 == node2.DPwMin):
						if(SPNLPpy_syntacticalGraphOperations.printVerbose):
							print("calculateMetricConnection: node1.lemma = ", node1.lemma, ", node2.lemma = ", node2.lemma)
						proximity = SPNLPpy_syntacticalGraphOperations.calculateProximityConnection(node1.w, node2.w)
						frequency = SPNLPpy_syntacticalGraphOperations.calculateFrequencyConnection(sentenceTreeNodeList, node1, node2)
						recency = SPNLPpy_syntacticalGraphOperations.calculateRecencyConnection(sentenceTreeNodeList, node1, node2, currentTime)	#minimise the difference in concept last access recency between left/right node
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
			print("generateSyntacticalTreeDependencyParserWordVectors: error connectionFound - check calculateMetricConnection parameters > 0.0, maxConnectionMetric = ", maxConnectionMetric)
			exit()

		if(SPNLPpy_syntacticalGraphOperations.printVerbose):
			print("create connection; w1 w2 = ", connectionNode1.w, " ", connectionNode2.w, ", connectionNode1.lemma connectionNode2.lemma = ", connectionNode1.lemma, " ", connectionNode2.lemma, ", metric = ", maxConnectionMetric)
		
		if(interpretRightNodeAsGovernor):
			connectionNode1Temp = connectionNode1
			connectionNode1 = connectionNode2
			connectionNode2 = connectionNode1Temp
		#CHECKTHIS limitation - infers directionality (source/target) of connection based on w1/w2 word order		
		#CHECKTHIS: always assume left to right directionality	
		#FUTURE: determine governor/dependent based on some other rule
		#interpret connectionNode1=connectionNodeGovernor, connectionNode2=connectionNodeDependent
		
		#primary vars;
		wordVector = SPNLPpy_syntacticalGraphOperations.getBranchWordVectorFromSourceNodes(connectionNode1, connectionNode2)

		#sentenceTreeArtificial vars;
		DPsubgraphSize = connectionNode1.DPsubgraphSize + connectionNode2.DPsubgraphSize
		conceptWordVector = np.add(connectionNode1.conceptWordVector, connectionNode2.conceptWordVector)
		conceptTime = connectionNode1.conceptTime + connectionNode2.conceptTime
		DPtreeLevel = max(connectionNode1.DPtreeLevel, (connectionNode2.DPtreeLevel+1))
		DPwMin = min(connectionNode1.DPwMin, connectionNode2.DPwMin)
		DPwMax = max(connectionNode1.DPwMax, connectionNode2.DPwMax)
		
		connectionNode1.wordVector = wordVector
		connectionNode1.DPsubgraphSize = DPsubgraphSize
		connectionNode1.conceptWordVector = conceptWordVector
		connectionNode1.conceptTime = conceptTime
		connectionNode1.DPtreeLevel = DPtreeLevel
		connectionNode1.DPwMin = DPwMin
		connectionNode1.DPwMax = DPwMax
				
		#connection vars;
		SPNLPpy_syntacticalGraphOperations.createGraphConnectionDP(connectionNode1, connectionNode2, addToConnectionsDictionary=False)
		connectivityStackNodeList.remove(connectionNode2)	#remove dependent from stack, every dependent can only have 1 governor

		if(len(connectivityStackNodeList) == 1):
			headNodeFound = True
			connectionNode1.graphNodeType = graphNodeTypeHead	#reference set delimiter (captures primary subject/action/object of sentence clause)
			graphHeadNode = connectionNode1
			
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
