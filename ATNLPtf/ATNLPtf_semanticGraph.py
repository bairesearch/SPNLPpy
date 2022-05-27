"""ATNLPtf_semanticGraph.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATNLPtf_main.py

# Usage:
see ATNLPtf_main.py

# Description:
ATNLP Semantic Graph - generate semantic graph/network based on a transformation of the ATNLP syntactical tree/graph

- semantic graph constructed based on GIA specification/implementation (ATNLPtf_semanticGraph is a minimal python implementation of GIA)
- GIAposRelTranslator applies a similar (axis orthogonal) transformation to a SANI graph to generate a GIA semantic graph (semantic relation/connection identification)
- ATNLP syntactical graph is not equivalent to a formal/strict syntax tree or SANI tree, so apply a custom transformation for semantic graph construction 

"""

import numpy as np
import spacy
from ATNLPtf_semanticNodeClass import *
from ATNLPtf_syntacticalNodeClass import *

supportMultiwordVerbsPrepositions = True

drawSyntacticalGraphTemporaryAfterRelationshipTransformation = True	#for visual debug after moveRelationshipSyntacticalNodes phase
if(drawSyntacticalGraphTemporaryAfterRelationshipTransformation):
	import ATNLPtf_syntacticalGraphDraw
	from ATNLPtf_semanticNodeClass import identifyEntityType
	

drawSemanticGraph = False
if(drawSemanticGraph):
	import ATNLPtf_semanticGraphDraw

def constructPOSdictionary():
	ATNLPtf_getAllPossiblePosTags.constructPOSdictionary()	#required for getKeypoints
 

def generateSemanticGraph(sentenceSyntacticalLeafNodeList, sentenceSyntacticalTreeNodeList, syntacticalGraphHeadNode):

	ATNLPtf_syntacticalGraphDraw.setColourSyntacticalNodes(True)	#always color nodes when generating semantic graph
	print("ATNLPtf_semanticGraph: ATNLPtf_syntacticalGraphDraw.drawSyntacticalGraphNodeColours = ", ATNLPtf_syntacticalGraphDraw.drawSyntacticalGraphNodeColours)
	
	if(drawSyntacticalGraphTemporaryAfterRelationshipTransformation):
		ATNLPtf_syntacticalGraphDraw.clearSyntacticalGraph()	
	if(drawSemanticGraph):
		ATNLPtf_semanticGraphDraw.clearSemanticGraph()
	
	sentenceSemanticNodeList = []
	if(supportMultiwordVerbsPrepositions):
		identifyMultiwordRelationshipLeafNodes(sentenceSyntacticalLeafNodeList)
	generateSemanticNodes(sentenceSemanticNodeList, sentenceSyntacticalLeafNodeList)		
	
	moveRelationshipSyntacticalNodes(sentenceSyntacticalLeafNodeList, sentenceSyntacticalTreeNodeList)
	#convertToSemanticGraph(sentenceSemanticNodeList, sentenceSyntacticalLeafNodeList, sentenceSyntacticalTreeNodeList)	#TODO

	if(drawSyntacticalGraphTemporaryAfterRelationshipTransformation):
		ATNLPtf_syntacticalGraphDraw.drawSyntacticalGraph(syntacticalGraphHeadNode)
				
	if(drawSyntacticalGraphTemporaryAfterRelationshipTransformation):
		ATNLPtf_syntacticalGraphDraw.displaySyntacticalGraph()	
	if(drawSemanticGraph):
		ATNLPtf_semanticGraphDraw.displaySemanticGraph()


def generateSemanticNodes(sentenceSemanticNodeList, sentenceSyntacticalLeafNodeList):
	for syntacticalNode in sentenceSyntacticalLeafNodeList:
		entityName = syntacticalNode.lemma 
		entityType = identifyEntityType(syntacticalNode)
		#syntacticalNode.colorHtml = ATNLPtf_semanticGraphDraw.getEntityNodeColour(entityType)
		#print("entityName = ", entityName, ", entityType = ", entityType)
		semanticNode = SemanticNode(syntacticalNode.instanceID, entityName, syntacticalNode.word, syntacticalNode.wordVector, entityType)
		sentenceSemanticNodeList.append(semanticNode)
		if(drawSemanticGraph):
			ATNLPtf_semanticGraphDraw.drawSemanticGraphNode(entityName)	#do not manually define a position (automatically generated)

def identifyMultiwordRelationshipLeafNodes(sentenceSyntacticalLeafNodeList):
	#in case ATNLPtf_syntacticalGraph.drawSyntacticalGraphNodeColours = False;
	for syntacticalNode in sentenceSyntacticalLeafNodeList:
		entityType = identifyEntityType(syntacticalNode)
		syntacticalNode.entityType = entityType
		
	#CHECKTHIS: currently only support 2 word multiword verbs/prepositions
	leafNodesToRemove = []
	leafNodesToAdd = []
	leafNodeInsertionIndex = -1
	for leafNodeIndex, leafNode in enumerate(sentenceSyntacticalLeafNodeList):
		if(not leafNode.multiwordLeafNode):
			currentBranchNode = leafNode.graphNodeTargetList[graphNodeTargetIndex]
			sourceNode1 = currentBranchNode.graphNodeSourceList[graphNodeSourceIndexFirst]
			sourceNode2 = currentBranchNode.graphNodeSourceList[graphNodeSourceIndexSecond]
			if((sourceNode1.graphNodeType == graphNodeTypeLeaf) and (sourceNode2.graphNodeType == graphNodeTypeLeaf)):
				#print("sourceNode1.entityType = ", sourceNode1.entityType)
				#print("sourceNode2.entityType = ", sourceNode2.entityType)
				if(isMultiwordRelationship(sourceNode1, sourceNode2)):
					currentBranchNode.entityType = sourceNode1.entityType	#CHECKTHIS: first word in phrasal verb/multiword preposition, auxiliary sequence
					print("multiword verbs/prepositions found")
					sourceNode1.multiwordLeafNode = True
					sourceNode2.multiwordLeafNode = True
					leafNodeInsertionIndex = leafNodeInsertionIndex
					leafNodesToRemove.append(sourceNode1)
					leafNodesToRemove.append(sourceNode2)
					leafNodesToAdd.append(currentBranchNode)
					
	for leafNode in leafNodesToRemove:
		sentenceSyntacticalLeafNodeList.remove(leafNode)
	for leafNode in leafNodesToAdd:
		sentenceSyntacticalLeafNodeList.insert(leafNodeInsertionIndex, leafNode)

def isMultiwordRelationship(sourceNode1, sourceNode2):
	relationshipEntity = False
	if(entityTypeIsRelationship(sourceNode1.entityType) and entityTypeIsRelationship(sourceNode2.entityType)):
		relationshipEntity = True
	else:
		for multiwordRelationshipSpecial in multiwordRelationshipSpecialList:
			if((sourceNode1.lemma == multiwordRelationshipSpecial[0]) and (sourceNode2.lemma == multiwordRelationshipSpecial[1])):
				relationshipEntity = True
				print("multiwordRelationshipSpecial found")
	return relationshipEntity
				
def moveRelationshipSyntacticalNodes(sentenceSyntacticalLeafNodeList, sentenceSyntacticalTreeNodeList):
				
	#CHECKTHIS: parse syntactical graph leafs from left to right;
	for leafNodeIndex, leafNode in enumerate(sentenceSyntacticalLeafNodeList):
		entityType = identifyEntityType(leafNode)
		if(entityTypeIsRelationship(entityType)):
			#print("entityTypeIsRelationship")
			#if first or last leaf in syntactical branch is relationship node (verb/proposition:action/condition) - transport the adjacent branch contents to the subject/object of the relationship node, and place the relationship node between the two branches;
			if(leafNode.sourceNodePosition == sourceNodePositionFirst):
				#the first leaf element in branch connects to entire previous branch
				foundBranchHead, branchHead, relationshipNode = findBranchHead(leafNode, branchEdgeFirstOrLast=True)		
			elif(leafNode.sourceNodePosition == sourceNodePositionSecond):
				#the last leaf element in branch connects to entire next branch
				foundBranchHead, branchHead, relationshipNode = findBranchHead(leafNode, branchEdgeFirstOrLast=False)	
			elif(leafNode.sourceNodePosition == sourceNodePositionUnknown):
				print("moveRelationshipSyntacticalNodes error: (leafNode.sourceNodePosition == sourceNodePositionUnknown)")
				exit()

			if(branchHead.graphNodeType == graphNodeTypeHead):
				#print("relationship = reference set (sentence) delimiter")
				relationshipNode.referenceSetDelimiter = True
			else:
				#print("relationship = subreference set (subsentence) delimiter")
				relationshipNode.subreferenceSetDelimiter = True

			branchHeadSourceFirstFound = False
			branchHeadSourceSecondFound = False
			if(foundBranchHead):
				#relationship found with subject and object
				branchHeadSourceFirstFound = True
				branchHeadSourceSecondFound = True
			else:
				#tree head found before finding final matched branch head
				if(entityIsRelationshipAction(entityType)):
					#relationship node found without either a subject or object
					if(leafNode.sourceNodePosition == sourceNodePositionFirst):
						#relationship node found without subject
						branchHeadSourceSecondFound = True
					elif(leafNode.sourceNodePosition == sourceNodePositionSecond):
						#relationship node found without object
						branchHeadSourceFirstFound = True
				elif(entityIsRelationshipCondition(entityType)):
					print("moveRelationshipSyntacticalNodes error: condition (preposition) entities require subject and object")
					exit()

			moveRelationshipSyntacticalNode(branchHead, relationshipNode)
											
def findBranchHead(leafNode, branchEdgeFirstOrLast):
	relationshipNode = leafNode
	relationshipNode.relationshipNodeMoved = True
	foundBranchHead = False
	branchHead = None
	branchSourceNodePositionIsEdge = True
	currentBranchNode = leafNode
	while(branchSourceNodePositionIsEdge):
		if(currentBranchNode.graphNodeType == graphNodeTypeHead):
			foundBranchHead = False	#tree head reached without appropriate branch/tree head identification
			branchHead = currentBranchNode
			branchSourceNodePositionIsEdge = False	#exit loop
		else: 
			currentBranchNode = currentBranchNode.graphNodeTargetList[graphNodeTargetIndex]
					
			if(currentBranchNode.sourceNodePosition != sourceNodePositionUnknown):
				if(branchEdgeFirstOrLast):
					currentBranchNode.lemma = currentBranchNode.lemma[len(relationshipNode.lemma):] #remove relationship from name
					if(currentBranchNode.sourceNodePosition != sourceNodePositionFirst):
						branchSourceNodePositionIsEdge = False
						branchHead = currentBranchNode.graphNodeTargetList[graphNodeTargetIndex]
						foundBranchHead = True
				else:
					currentBranchNode.lemma = currentBranchNode.lemma[:-len(relationshipNode.lemma)] #remove relationship from name
					if(currentBranchNode.sourceNodePosition != sourceNodePositionSecond):
						branchSourceNodePositionIsEdge = False
						branchHead = currentBranchNode.graphNodeTargetList[graphNodeTargetIndex]
						foundBranchHead = True
			
	return foundBranchHead, branchHead, relationshipNode

def moveRelationshipSyntacticalNode(branchHead, relationshipLeafNode):	
	relationshipLeafNode.graphNodeType = graphNodeTypeRelationship	
	removeNodeTargetConnections(relationshipLeafNode)
	addConnectionToNodeTargets(relationshipLeafNode, branchHead)
	addConnectionToNodeSources(branchHead, relationshipLeafNode)	
				
		

