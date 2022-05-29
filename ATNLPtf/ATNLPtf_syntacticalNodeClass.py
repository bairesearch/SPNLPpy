"""ATNLPtf_syntacticalNodeClass.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATNLPtf_main.py

# Usage:
see ATNLPtf_main.py

# Description:
ATNLP Syntactical Node Class

"""

import numpy as np

graphNodeTypeUnknown = 0
graphNodeTypeLeaf = 1	#base/input neuron (tree branch leaf)
graphNodeTypeBranch = 2	#hidden neuron (tree branch head: contains a tree branch contents)
graphNodeTypeHead = 3	#top/output neuron (tree head: contains tree contents)
graphNodeTypeRelationship = 4	#artificial node type for relationship (action/condition) after moving from leaf node to branch head

sourceNodePositionUnknown = 0
sourceNodePositionFirst = 1
sourceNodePositionSecond = 2

graphNodeTargetIndex = 0	#should only contain one element
graphNodeSourceIndexFirst = 0	#should only contain 2 elements
graphNodeSourceIndexSecond = 1	#should only contain 2 elements

#FUTURE: move these to ATNLPtf_syntacticalNodeClass;
class SyntacticalNode:
	def __init__(self, instanceID, word, lemma, wordVector, posTag, nodeGraphType, activationTime, subgraphSize, conceptWordVector, conceptTime, w, wMin, wMax, treeLevel, sentenceIndex):
		#primary vars;
		self.instanceID = instanceID
		self.word = word
		self.lemma = lemma
		self.wordVector = wordVector	#numpy array
		self.posTag = posTag	#nlp in context prediction only (not certain)
		self.graphNodeType = nodeGraphType
		self.activationTime = activationTime	#last activation time (used to calculate recency)
		self.constituencyParserLabel = None	#not used (stored for reference)
		
		#sentenceTreeArtificial vars;
		self.subgraphSize = subgraphSize	#used to normalise wordVector/conceptTime for hidden nodes
		self.conceptWordVector = conceptWordVector	#requires /subgraphSize
		self.conceptTime = conceptTime	#requires /subgraphSize
		self.w = w #temporary sentence word index (used for reference resolution only)
		self.wMin = wMin	#temporary sentence word index (used for reference resolution only) - min of all hidden nodes
		self.wMax = wMax	#temporary sentence word index (used for reference resolution only) - max of all hidden nodes
		self.treeLevel = treeLevel
		self.sentenceIndex = sentenceIndex
		#self.referenceSentence = False	#temporary flag: node has been reference by current sentence (used for reference resolution only)
		
		#connection vars;
		self.graphNodeTargetList = []	#should only contain one element
		self.graphNodeSourceList = []
		#self.graphNodeTargetDict = {}	#dict indexed by lemma, every entry is a dictionary of SyntacticalNode instances indexed by instanceID 	#for optimised lookup by concept
		#self.graphNodeSourceDict = {}	#dict indexed by lemma, every entry is a dictionary of SyntacticalNode instances indexed by instanceID	#for optimised lookup by concept
		#self.foundRecentIndex = False	#temporary var (indicates referencing a previously declared instance in the article)
		self.sourceNodePosition = sourceNodePositionUnknown	#for leaf nodes only 
		
		#intermediary vars for semantic graph generation;
		self.entityType = -1	#temp #GIA_ENTITY_TYPE_UNDEFINED
		self.multiwordLeafNode = False
		self.referenceSetDelimiter = False	#if entityType IsRelationship only
		self.subreferenceSetDelimiter = False	#if entityType IsRelationship only
		
		#temporary graph draw variables
		self.drawn = False
		
def addConnectionToNodeTargets(node, nodeToConnect):
	node.graphNodeTargetList.append(nodeToConnect)
	#addInstanceNodeToDictionary(node.graphNodeTargetDict, nodeToConnect.lemma, nodeToConnect.instanceID, nodeToConnect)

def addConnectionToNodeSources(node, nodeToConnect):
	node.graphNodeSourceList.append(nodeToConnect)
	#addInstanceNodeToDictionary(node.graphNodeSourceDict, nodeToConnect.lemma, nodeToConnect.instanceID, nodeToConnect)
		
def removeNodeConnections(node1):	
	removeNodeSourceConnections(node1)
	removeNodeTargetConnections(node1)

def removeNodeSourceConnections(node1):	
	for node1sourceIndex, node1source in enumerate(node1.graphNodeSourceList):
		for node1sourceTargetIndex, node1sourceTarget in enumerate(node1source.graphNodeTargetList):
			if(node1sourceTarget == node1):
				node1sourceTargetIndexDel = node1sourceTargetIndex
		del node1source.graphNodeTargetList[node1sourceTargetIndexDel]
	node1.graphNodeSourceList.clear()

def removeNodeTargetConnections(node1):	
	for node1targetIndex, node1target in enumerate(node1.graphNodeTargetList):
		for node1targetSourceIndex, node1targetSource in enumerate(node1target.graphNodeSourceList):
			if(node1targetSource == node1):
				node1targetSourceIndexDel = node1targetSourceIndex		
		del node1target.graphNodeSourceList[node1targetSourceIndexDel]
	node1.graphNodeTargetList.clear()
