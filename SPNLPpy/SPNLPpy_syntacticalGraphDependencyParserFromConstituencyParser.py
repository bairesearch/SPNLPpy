"""SPNLPpy_syntacticalGraphDependencyParserFromConstituencyParser.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SPNLPpy_main.py

# Usage:
see SPNLPpy_main.py

# Description:
SPNLP Syntactical Graph Dependency Parser (DP) Word Vectors - generate dependency parse tree/graph from constituency parse tree (based on adjacent constituency branch word frequency heuristics)
 
Preconditions: assumes syntactical/constituency graph/tree already generated

"""

import numpy as np
import spacy
import ANNtf2_loadDataset
from SPNLPpy_syntacticalNodeClass import *
import SPNLPpy_syntacticalGraphOperations

calibrateConnectionMetricParameters = True

primarySourceNodeDetectionCompareBranchHeadOrAdjacentBranchWordVector = False

#experimental
def generateSyntacticalTreeDependencyParserFromConstituencyParser(sentenceIndex, sentenceLeafNodeList, sentenceTreeNodeList, syntacticalGraphNodeDictionary, CPgraphHeadNode, performIntermediarySyntacticalTransformation):
	
	for leafNode in sentenceLeafNodeList:
		identifyPrimarySourceNodeSentence(leafNode, performIntermediarySyntacticalTransformation)
	for leafNode in sentenceLeafNodeList:
		recordPrimaryLeafNodeSentence(leafNode, leafNode, performIntermediarySyntacticalTransformation)
	for leafNode in sentenceLeafNodeList:
		formDependencyRelations(leafNode, performIntermediarySyntacticalTransformation)
	
	print("CPgraphHeadNode.CPprimaryLeafNode.lemma = ", CPgraphHeadNode.CPprimaryLeafNode.lemma)
	
	DPgraphHeadNode = CPgraphHeadNode.CPprimaryLeafNode

	calculateNodeTreeLevelSentence(DPgraphHeadNode)
	
	return DPgraphHeadNode

def identifyPrimarySourceNodeSentence(node, performIntermediarySyntacticalTransformation):	
	#print("identifyPrimarySourceNodeSentence: node = ", node.lemma)

	if(len(node.CPgraphNodeTargetList) > 0):
		for targetNode in node.CPgraphNodeTargetList:
			CPisPrimarySourceNode = identifyPrimarySourceNode(node, targetNode, performIntermediarySyntacticalTransformation)
			if(CPisPrimarySourceNode):
				identifyPrimarySourceNodeSentence(targetNode, performIntermediarySyntacticalTransformation)
	else:
		node.CPisPrimarySourceNode = True
	
def identifyPrimarySourceNode(node, targetNode, performIntermediarySyntacticalTransformation):
	
	#print("identifyPrimarySourceNode: node = ", node.lemma, ", targetNode = ", targetNode.lemma)
	if(primarySourceNodeDetectionCompareBranchHeadOrAdjacentBranchWordVector):
		comparisonBranchFound, comparisonBranchNode = identifyBranchHeadNode(node, targetNode)
	else:
		comparisonBranchFound, comparisonBranchNode = identifyAdjacentBranchNode(node, targetNode)
	if(comparisonBranchFound):
		#print("comparisonBranchFound")
		adjacentNodeFound, adjacentNode = identifyAdjacentNode(node, targetNode)
		if(adjacentNodeFound):
			#CHECKTHIS #if parentX child1 [leaf] node is more associated with adjacent branch than parentX child2 [leaf] node than it becomes the primary node
			comparison1 = SPNLPpy_syntacticalGraphOperations.compareWordVectors(node.wordVector, comparisonBranchNode.wordVector)
			comparison2 = SPNLPpy_syntacticalGraphOperations.compareWordVectors(adjacentNode.wordVector, comparisonBranchNode.wordVector)
			#print("node.wordVector = ", node.wordVector) 
			#print("adjacentNode.wordVector = ", adjacentNode.wordVector) 
			#print("comparisonBranchNode.wordVector = ", comparisonBranchNode.wordVector) 
			#print("comparison1 = ", comparison1) 
			#print("comparison2 = ", comparison2)
			if(comparison1 < comparison2):
				node.CPisPrimarySourceNode = True
			elif(comparison1 == comparison2):
				#print("SPNLPpy_syntacticalGraphConstituencyParserWordVectors: identifyPrimarySourceNode warning: comparison1 == comparison2")
				if(node.CPsourceNodePosition == sourceNodePositionFirst):
					node.CPisPrimarySourceNode = True
		else:
			node.CPisPrimarySourceNode = True
	else:
		#print("!comparisonBranchFound")
		#targetNode is graphNodeHead; set first node in targetNode.source as governor	#CHECKTHIS
		#print("node.CPsourceNodePosition = ", node.CPsourceNodePosition)
		if(node.CPsourceNodePosition == sourceNodePositionFirst):
			node.CPisPrimarySourceNode = True
	return node.CPisPrimarySourceNode
			
def identifyAdjacentNode(node, targetNode):
	adjacentNodeFound = False
	adjacentNode = None
	for nodeTemp in targetNode.CPgraphNodeSourceList:
		if(nodeTemp is not node):
			adjacentNodeFound = True
			adjacentNode = nodeTemp
			
	if(not adjacentNodeFound):
		if(performIntermediarySyntacticalTransformation):
			pass
			#this case might be caused by SPNLPpy_syntacticalGraphIntermediaryTransformation (creates branches/hiddenNodes with single target and source)	
		else:
			print("identifyAdjacentNode error: !adjacentNodeFound")
			exit()
				
	return adjacentNodeFound, adjacentNode

def identifyBranchHeadNode(node, targetNode):
	branchHeadFound = False
	branchHeadNode = None
	for branchHead in targetNode.CPgraphNodeTargetList:
		branchHeadFound = True
		branchHeadNode = branchHead
	return branchHeadFound, branchHeadNode
						
def identifyAdjacentBranchNode(node, targetNode):
	adjacentBranchFound = False
	adjacentBranchNode = None
	for branchHead in targetNode.CPgraphNodeTargetList:
		for adjacentBranch in branchHead.CPgraphNodeSourceList:
			adjacentBranchFound = True
			adjacentBranchNode = adjacentBranch
	return adjacentBranchFound, adjacentBranchNode
	
def recordPrimaryLeafNodeSentence(node, CPprimaryLeafNode, performIntermediarySyntacticalTransformation):
	node.CPprimaryLeafNode = CPprimaryLeafNode
	#print("recordPrimaryLeafNodeSentence: node = ", node.lemma, ", CPprimaryLeafNode = ", CPprimaryLeafNode.lemma)
	if(node.CPisPrimarySourceNode):
		for targetNode in node.CPgraphNodeTargetList:
			recordPrimaryLeafNodeSentence(targetNode, CPprimaryLeafNode, performIntermediarySyntacticalTransformation)
			#adjacentNodeFound, adjacentNode = identifyAdjacentNode(node, targetNode)


def formDependencyRelations(node, performIntermediarySyntacticalTransformation):	
	#print("formDependencyRelations: node = ", node.lemma)
	for targetNode in node.CPgraphNodeTargetList:
		if(node.CPisPrimarySourceNode):
			adjacentNodeFound, adjacentNode = identifyAdjacentNode(node, targetNode)
			#print("\t formDependencyRelations: adjacentNode = ", adjacentNode.lemma)
			if(adjacentNodeFound):
				node.CPprimaryLeafNode.DPdependentList.append(adjacentNode.CPprimaryLeafNode)	
				adjacentNode.CPprimaryLeafNode.DPgovernorList.append(node.CPprimaryLeafNode)
			formDependencyRelations(targetNode, performIntermediarySyntacticalTransformation)

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


			
