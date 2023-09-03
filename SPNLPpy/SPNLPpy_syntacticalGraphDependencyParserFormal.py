"""SPNLPpy_syntacticalGraphDependencyParserFormal.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SPNLPpy_main.py

# Usage:
see SPNLPpy_main.py

# Description:
SPNLP Syntactical Graph Dependency Parser (DP) Formal - external python dependency parse tree generation

Preconditions: assumes leaf nodes already generated

"""

import numpy as np
import spacy
from SPNLPpy_syntacticalNodeClass import *

dependencyTreeRootNodeName = "ROOT"

def generateSyntacticalTreeDependencyParserFormal(sentenceIndex, tokenisedSentence, sentenceLeafNodeList, sentenceTreeNodeList, syntacticalGraphNodeDictionary):

	graphHeadNode = None
	
	syntacticalDependencyRelationList = []

	for w, tokenDependent in enumerate(tokenisedSentence):
		#print(tokenDependent.text, tokenDependent.tag_, tokenDependent.head.text, tokenDependent.dep_)
		#print("sentenceLeafNodeList[w].lemma = ", sentenceLeafNodeList[w].lemma)
		#print("w = ", w)
		#print("token.i = ", tokenDependent.i)
		leafNodeDependent = sentenceLeafNodeList[tokenDependent.i]
		dependencyRelationLabel = tokenDependent.dep_
		if(tokenDependent.dep_ == dependencyTreeRootNodeName):
			graphHeadNode = leafNodeDependent
		else:
			tokenGovernor = tokenDependent.head
			leafNodeGovernor = sentenceLeafNodeList[tokenGovernor.i]
			leafNodeDependent.DPgovernorList.append(leafNodeGovernor)
			leafNodeGovernor.DPdependentList.append(leafNodeDependent)
			leafNodeDependent.DPdependencyRelationLabelList.append(dependencyRelationLabel)

	
	if(graphHeadNode is None):
		print("generateSyntacticalDependencyRelations error: graphHeadNode (ROOT) not found")
		exit()
		
	calculateNodeTreeLevelSentence(graphHeadNode)

	return graphHeadNode

def calculateNodeTreeLevelSentence(syntacticalGraphNode):	
	treeLevel = calculateNodeTreeLevel(syntacticalGraphNode, 0)
	#print("treeLevel = ", treeLevel)
	syntacticalGraphNode.DPtreeLevel = treeLevel
	for sourceNode in syntacticalGraphNode.DPdependentList:
		calculateNodeTreeLevelSentence(sourceNode)

def calculateNodeTreeLevel(syntacticalGraphNode, treeLevel):	
	maxTreeLevelBranch = treeLevel
	for sourceNode in syntacticalGraphNode.DPdependentList:
		treeLevelTemp = calculateNodeTreeLevel(sourceNode, treeLevel+1)
		if(treeLevelTemp > maxTreeLevelBranch):
			maxTreeLevelBranch = treeLevelTemp
	return maxTreeLevelBranch

