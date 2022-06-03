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
SPNLP Syntactical Graph Dependency Parser Word Vectors - identify syntactical graph dependency relation governor/dependents using input vectors

Preconditions: assumes syntactical/constituency graph/tree already generated

"""

import numpy as np
import spacy
import ANNtf2_loadDataset
from SPNLPpy_syntacticalNodeClass import *
import SPNLPpy_syntacticalGraphOperations

calibrateConnectionMetricParameters = True

#experimental
def generateSyntacticalTreeDependencyParserWordVectors(sentenceIndex, sentenceLeafNodeList, sentenceTreeNodeList, syntacticalGraphNodeDictionary, constituencyParserGraphHeadNode, performIntermediarySyntacticalTransformation):
	
	for leafNode in sentenceLeafNodeList:
		identifyPrimarySourceNodeSentence(leafNode, performIntermediarySyntacticalTransformation)
	for leafNode in sentenceLeafNodeList:
		recordPrimaryLeafNodeSentence(leafNode, leafNode, performIntermediarySyntacticalTransformation)
	for leafNode in sentenceLeafNodeList:
		formDependencyRelations(leafNode, performIntermediarySyntacticalTransformation)
	
	print("constituencyParserGraphHeadNode.primaryLeafNode.lemma = ", constituencyParserGraphHeadNode.primaryLeafNode.lemma)
	
	dependencyParserGraphHeadNode = constituencyParserGraphHeadNode.primaryLeafNode

	calculateNodeTreeLevelSentence(dependencyParserGraphHeadNode)
	
	return dependencyParserGraphHeadNode

def identifyPrimarySourceNodeSentence(node, performIntermediarySyntacticalTransformation):	
	#print("identifyPrimarySourceNodeSentence: node = ", node.lemma)

	if(len(node.graphNodeTargetList) > 0):
		for targetNode in node.graphNodeTargetList:
			isPrimarySourceNode = identifyPrimarySourceNode(node, targetNode, performIntermediarySyntacticalTransformation)
			if(isPrimarySourceNode):
				identifyPrimarySourceNodeSentence(targetNode, performIntermediarySyntacticalTransformation)
	else:
		node.isPrimarySourceNode = True
	
def identifyPrimarySourceNode(node, targetNode, performIntermediarySyntacticalTransformation):
	
	#print("identifyPrimarySourceNode: node = ", node.lemma, ", targetNode = ", targetNode.lemma)
	adjacentBranchFound, adjacentBranchNode = identifyAdjacentBranchNode(node, targetNode)
	if(adjacentBranchFound):
		#print("adjacentBranchFound")
		adjacentNodeFound, adjacentNode = identifyAdjacentNode(node, targetNode)
		if(adjacentNodeFound):
			#CHECKTHIS #if parentX child1 [leaf] node is more associated with adjacent branch than parentX child2 [leaf] node than it becomes the primary node
			comparison1 = SPNLPpy_syntacticalGraphOperations.compareWordVectors(node.wordVector, adjacentBranchNode.wordVector)
			comparison2 = SPNLPpy_syntacticalGraphOperations.compareWordVectors(adjacentNode.wordVector, adjacentBranchNode.wordVector)
			#print("node.wordVector = ", node.wordVector) 
			#print("adjacentNode.wordVector = ", adjacentNode.wordVector) 
			#print("adjacentBranchNode.wordVector = ", adjacentBranchNode.wordVector) 
			#print("comparison1 = ", comparison1) 
			#print("comparison2 = ", comparison2)
			if(comparison1 < comparison2):
				node.isPrimarySourceNode = True
			elif(comparison1 == comparison2):
				#print("SPNLPpy_syntacticalGraphConstituencyParserWordVectors: identifyPrimarySourceNode warning: comparison1 == comparison2")
				if(node.sourceNodePosition == sourceNodePositionFirst):
					node.isPrimarySourceNode = True
		else:
			node.isPrimarySourceNode = True
	else:
		#print("!adjacentBranchFound")
		#targetNode is graphNodeHead; set first node in targetNode.source as governor	#CHECKTHIS
		#print("node.sourceNodePosition = ", node.sourceNodePosition)
		if(node.sourceNodePosition == sourceNodePositionFirst):
			node.isPrimarySourceNode = True
	return node.isPrimarySourceNode
			
def identifyAdjacentNode(node, targetNode):
	adjacentNodeFound = False
	adjacentNode = None
	for nodeTemp in targetNode.graphNodeSourceList:
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
					
def identifyAdjacentBranchNode(node, targetNode):
	adjacentBranchFound = False
	adjacentBranchNode = None
	for branchHead in targetNode.graphNodeTargetList:
		for adjacentBranch in branchHead.graphNodeSourceList:
			adjacentBranchFound = True
			adjacentBranchNode = adjacentBranch
	return adjacentBranchFound, adjacentBranchNode
	
def recordPrimaryLeafNodeSentence(node, primaryLeafNode, performIntermediarySyntacticalTransformation):
	node.primaryLeafNode = primaryLeafNode
	#print("recordPrimaryLeafNodeSentence: node = ", node.lemma, ", primaryLeafNode = ", primaryLeafNode.lemma)
	if(node.isPrimarySourceNode):
		for targetNode in node.graphNodeTargetList:
			recordPrimaryLeafNodeSentence(targetNode, primaryLeafNode, performIntermediarySyntacticalTransformation)
			#adjacentNodeFound, adjacentNode = identifyAdjacentNode(node, targetNode)


def formDependencyRelations(node, performIntermediarySyntacticalTransformation):	
	#print("formDependencyRelations: node = ", node.lemma)
	for targetNode in node.graphNodeTargetList:
		if(node.isPrimarySourceNode):
			adjacentNodeFound, adjacentNode = identifyAdjacentNode(node, targetNode)
			#print("\t formDependencyRelations: adjacentNode = ", adjacentNode.lemma)
			if(adjacentNodeFound):
				node.primaryLeafNode.dependencyParserDependentList.append(adjacentNode.primaryLeafNode)	
				adjacentNode.primaryLeafNode.dependencyParserGovernorList.append(node.primaryLeafNode)	
			formDependencyRelations(targetNode, performIntermediarySyntacticalTransformation)

def calculateNodeTreeLevelSentence(syntacticalGraphNode):	
	treeLevel = calculateNodeTreeLevel(syntacticalGraphNode, 0)
	#print("treeLevel = ", treeLevel)
	syntacticalGraphNode.dependencyParserTreeLevel = treeLevel
	for sourceNode in syntacticalGraphNode.dependencyParserDependentList:
		calculateNodeTreeLevelSentence(sourceNode)

def calculateNodeTreeLevel(syntacticalGraphNode, treeLevel):	
	maxTreeLevelBranch = treeLevel
	for sourceNode in syntacticalGraphNode.dependencyParserDependentList:
		treeLevelTemp = calculateNodeTreeLevel(sourceNode, treeLevel+1)
		if(treeLevelTemp > maxTreeLevelBranch):
			maxTreeLevelBranch = treeLevelTemp
	return maxTreeLevelBranch


			
