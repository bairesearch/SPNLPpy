"""ATNLPtf_syntacticalGraphConstituencyParserWordVectors.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATNLPtf_main.py

# Usage:
see ATNLPtf_main.py

# Description:
ATNLP Syntactical Graph Constituency Parser Word Vectors - generate syntactical tree/graph using input vectors (based on word proximity, frequency, and recency heuristics)

Preconditions: assumes leaf nodes already generated

"""

import numpy as np
import spacy
spacyWordVectorGenerator = spacy.load('en_core_web_md')	#spacy.load('en_core_web_lg')
import ANNtf2_loadDataset
from ATNLPtf_syntacticalNodeClass import *
import ATNLPtf_syntacticalGraphOperations

calibrateConnectionMetricParameters = True

def generateSyntacticalTreeConstituencyParserWordVectors(sentenceIndex, sentenceLeafNodeList, sentenceTreeNodeList, connectivityStackNodeList, graphNodeDictionary):

	currentTime = ATNLPtf_syntacticalGraphOperations.calculateActivationTime(sentenceIndex)

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
						if(ATNLPtf_syntacticalGraphOperations.printVerbose):
							print("calculateMetricConnection: node1.lemma = ", node1.lemma, ", node2.lemma = ", node2.lemma)
						proximity = ATNLPtf_syntacticalGraphOperations.calculateProximityConnection(node1.w, node2.w)
						frequency = ATNLPtf_syntacticalGraphOperations.calculateFrequencyConnection(sentenceTreeNodeList, node1, node2)
						recency = ATNLPtf_syntacticalGraphOperations.calculateRecencyConnection(sentenceTreeNodeList, node1, node2, currentTime)	#minimise the difference in concept last access recency between left/right node
						connectionMetric = ATNLPtf_syntacticalGraphOperations.calculateMetricConnection(proximity, frequency, recency)
						if(connectionMetric > ATNLPtf_syntacticalGraphOperations.metricThresholdToCreateConnection):
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

		if(ATNLPtf_syntacticalGraphOperations.printVerbose):
			print("create connection; w1 w2 = ", connectionNode1.w, " ", connectionNode2.w, ", connectionNode1.lemma connectionNode2.lemma = ", connectionNode1.lemma, " ", connectionNode2.lemma, ", metric = ", maxConnectionMetric)

		#CHECKTHIS limitation - infers directionality (source/target) of connection based on w1/w2 word order		
		connectionDirection = True	#CHECKTHIS: always assume left to right directionality

		#primary vars;
		word = connectionNode1.word + connectionNode2.word
		lemma = connectionNode1.lemma + connectionNode2.lemma
		wordVector = ATNLPtf_syntacticalGraphOperations.getBranchWordVectorFromSourceNodes(connectionNode1, connectionNode2)
		posTag = None
		activationTime = ATNLPtf_syntacticalGraphOperations.calculateActivationTime(sentenceIndex)	#mean([connectionNode1.activationTime, connectionNode2.activationTime]) 
		nodeGraphType = graphNodeTypeBranch

		#sentenceTreeArtificial vars;
		subgraphSize = connectionNode1.subgraphSize + connectionNode2.subgraphSize + 1
		conceptWordVector = np.add(connectionNode1.conceptWordVector, connectionNode2.conceptWordVector)
		conceptTime = connectionNode1.conceptTime + connectionNode2.conceptTime
		treeLevel = max(connectionNode1.treeLevel, connectionNode2.treeLevel) + 1
		w = ATNLPtf_syntacticalGraphOperations.mean([connectionNode1.w, connectionNode2.w])
		wMin = min(connectionNode1.wMin, connectionNode2.wMin)
		wMax = max(connectionNode1.wMax, connectionNode2.wMax)

		instanceID = ATNLPtf_syntacticalGraphOperations.getNewInstanceID(graphNodeDictionary, lemma)
		hiddenNode = SyntacticalNode(instanceID, word, lemma, wordVector, posTag, nodeGraphType, currentTime, subgraphSize, conceptWordVector, conceptTime, w, wMin, wMax, treeLevel, sentenceIndex)
		ATNLPtf_syntacticalGraphOperations.addInstanceNodeToGraph(graphNodeDictionary, lemma, instanceID, hiddenNode)
		
		#connection vars;
		ATNLPtf_syntacticalGraphOperations.createGraphConnectionWrapper(hiddenNode, connectionNode1, connectionNode2, connectionDirection, addToConnectionsDictionary=False)
		sentenceTreeNodeList.append(hiddenNode)
		connectivityStackNodeList.remove(connectionNode1)
		connectivityStackNodeList.remove(connectionNode2)
		connectivityStackNodeList.append(hiddenNode)

		if(len(connectivityStackNodeList) == 1):
			headNodeFound = True
			hiddenNode.graphNodeType = graphNodeTypeHead	#reference set delimiter (captures primary subject/action/object of sentence clause)
			graphHeadNode = hiddenNode
			
	if(calibrateConnectionMetricParameters):
		proximityMinMeanMax = ATNLPtf_syntacticalGraphOperations.minMeanMaxList(proximityList)
		frequencyMinMeanMax = ATNLPtf_syntacticalGraphOperations.minMeanMaxList(frequencyList)
		recencyMinMeanMax = ATNLPtf_syntacticalGraphOperations.minMeanMaxList(recencyList)
		metricMinMeanMax = ATNLPtf_syntacticalGraphOperations.minMeanMaxList(metricList)
		print("proximityMinMeanMax = ", proximityMinMeanMax)
		print("frequencyMinMeanMax = ", frequencyMinMeanMax)
		print("recencyMinMeanMax = ", recencyMinMeanMax)
		print("metricMinMeanMax = ", metricMinMeanMax)
			
	return graphHeadNode

