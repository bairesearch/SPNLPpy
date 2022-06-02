"""SPNLPpy_semanticGraphIntermediaryTransformation.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SPNLPpy_main.py

# Usage:
see SPNLPpy_main.py

# Description:
SPNLP Semantic Graph Intermediary Transformation - perform intermediary semantic transformation of syntactical graph (move relationship nodes)

"""

import numpy as np
import spacy
from SPNLPpy_syntacticalNodeClass import *
import SPNLPpy_semanticNodeClass

supportMultiwordVerbsPrepositions = True

drawSyntacticalGraphTemporaryAfterRelationshipTransformation = True	#for visual debug after moveRelationshipSyntacticalNodes phase
if(drawSyntacticalGraphTemporaryAfterRelationshipTransformation):
	import SPNLPpy_syntacticalGraphDraw as SPNLPpy_syntacticalGraphDrawSentence
 

def performIntermediarySemanticTransformation(parserType, sentenceSyntacticalLeafNodeList, sentenceSyntacticalTreeNodeList, syntacticalGraphHeadNode):

	SPNLPpy_syntacticalGraphDrawSentence.setColourSyntacticalNodes(True)	#always color nodes when generating intermedary transformation graph
	print("SPNLPpy_semanticGraphIntermediaryTransformation: SPNLPpy_syntacticalGraphDrawSentence.drawSyntacticalGraphNodeColours = ", SPNLPpy_syntacticalGraphDrawSentence.drawSyntacticalGraphNodeColours)
	
	if(drawSyntacticalGraphTemporaryAfterRelationshipTransformation):
		SPNLPpy_syntacticalGraphDrawSentence.clearSyntacticalGraph()

	identifyEntityTypes(sentenceSyntacticalLeafNodeList)		
	
	if(parserType == "constituencyParserWordVector"):
		sentenceSyntacticalLeafNodeListMultiwords = list(sentenceSyntacticalLeafNodeList)

		if(supportMultiwordVerbsPrepositions):
			identifyMultiwordRelationshipLeafNodes(sentenceSyntacticalLeafNodeListMultiwords)

		moveRelationshipSyntacticalNodes(sentenceSyntacticalLeafNodeListMultiwords, sentenceSyntacticalTreeNodeList)
	elif(parserType == "constituencyParserFormal"):
		pass
		
	#TODO: detect determiners and substance/quality structure
	
	if(drawSyntacticalGraphTemporaryAfterRelationshipTransformation):
		SPNLPpy_syntacticalGraphDrawSentence.drawSyntacticalGraphSentence(syntacticalGraphHeadNode)
		print("SPNLPpy_syntacticalGraphDrawSentence.displaySyntacticalGraph()")
		SPNLPpy_syntacticalGraphDrawSentence.displaySyntacticalGraph()


def identifyEntityTypes(sentenceSyntacticalLeafNodeList):
	for syntacticalNode in sentenceSyntacticalLeafNodeList:
		entityName = syntacticalNode.lemma 
		entityType = SPNLPpy_semanticNodeClass.identifyEntityType(syntacticalNode)
		syntacticalNode.entityType = entityType

def identifyMultiwordRelationshipLeafNodes(sentenceSyntacticalLeafNodeList):
	#in case SPNLPpy_syntacticalGraph.drawSyntacticalGraphNodeColours = False;
	for syntacticalNode in sentenceSyntacticalLeafNodeList:
		entityType = SPNLPpy_semanticNodeClass.identifyEntityType(syntacticalNode)
		syntacticalNode.entityType = entityType
		
	#CHECKTHIS: currently only support 2 word multiword verbs/prepositions
	leafNodesToRemove = []
	leafNodesToAdd = []
	leafNodeInsertionIndex = -1
	for leafNodeIndex, leafNode in enumerate(sentenceSyntacticalLeafNodeList):
		if(not leafNode.multiwordLeafNode):
			#print("leafNode = ", leafNode.lemma)
			currentBranchNode = leafNode.graphNodeTargetList[graphNodeTargetIndex]
			sourceNode1 = currentBranchNode.graphNodeSourceList[graphNodeSourceIndexFirst]
			sourceNode2 = currentBranchNode.graphNodeSourceList[graphNodeSourceIndexSecond]
			if((sourceNode1.graphNodeType == graphNodeTypeLeaf) and (sourceNode2.graphNodeType == graphNodeTypeLeaf)):
				#print("sourceNode1.entityType = ", sourceNode1.entityType)
				#print("sourceNode2.entityType = ", sourceNode2.entityType)
				if(isMultiwordRelationship(sourceNode1, sourceNode2)):
					currentBranchNode.entityType = sourceNode1.entityType	#CHECKTHIS: first word in phrasal verb/multiword preposition, auxiliary sequence
					#print("multiword verbs/prepositions found")
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
	if(SPNLPpy_semanticNodeClass.entityTypeIsRelationship(sourceNode1.entityType) and SPNLPpy_semanticNodeClass.entityTypeIsRelationship(sourceNode2.entityType)):
		relationshipEntity = True
	else:
		for multiwordRelationshipSpecial in SPNLPpy_semanticNodeClass.multiwordRelationshipSpecialList:
			if((sourceNode1.lemma == multiwordRelationshipSpecial[0]) and (sourceNode2.lemma == multiwordRelationshipSpecial[1])):
				relationshipEntity = True
				print("multiwordRelationshipSpecial found")
	return relationshipEntity
				
def moveRelationshipSyntacticalNodes(sentenceSyntacticalLeafNodeList, sentenceSyntacticalTreeNodeList):
				
	#CHECKTHIS: parse syntactical graph leafs from left to right;
	for leafNodeIndex, leafNode in enumerate(sentenceSyntacticalLeafNodeList):
		entityType = SPNLPpy_semanticNodeClass.identifyEntityType(leafNode)
		if(SPNLPpy_semanticNodeClass.entityTypeIsRelationship(entityType)):
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
				if(SPNLPpy_semanticNodeClass.entityIsRelationshipAction(entityType)):
					#relationship node found without either a subject or object
					if(leafNode.sourceNodePosition == sourceNodePositionFirst):
						#relationship node found without subject
						branchHeadSourceSecondFound = True
					elif(leafNode.sourceNodePosition == sourceNodePositionSecond):
						#relationship node found without object
						branchHeadSourceFirstFound = True
				elif(SPNLPpy_semanticNodeClass.entityIsRelationshipCondition(entityType)):
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
				
		

