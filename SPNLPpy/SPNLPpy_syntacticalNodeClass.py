"""SPNLPpy_syntacticalNodeClass.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SPNLPpy_main.py

# Usage:
see SPNLPpy_main.py

# Description:
SPNLP Syntactical Node Class

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

syntacticalGraphTypeUnknown = 0
syntacticalGraphTypeConstituencyTree = 1
syntacticalGraphTypeDependencyTree = 2
syntacticalGraphTypeAcyclic = 3

class SyntacticalNode:
	def __init__(self, instanceID, word, lemma, wordVector, posTag, nodeGraphType, activationTime, CPsubgraphSize, conceptWordVector, conceptTime, w, CPwMin, CPwMax, CPtreeLevel, sentenceIndex):
		#primary vars;
		self.instanceID = instanceID
		self.word = word
		self.lemma = lemma
		self.wordVector = wordVector	#numpy array
		self.posTag = posTag	#nlp in context prediction only (not certain)
		self.graphNodeType = nodeGraphType
		self.activationTime = activationTime	#last activation time (used to calculate recency)
		
		#sentenceTreeArtificial vars (for sentence graph only, do not generalise to network graph);
		self.CPlabel = None	#not used (stored for reference)
		self.CPsubgraphSize = CPsubgraphSize	#for constituencyParser	#used to normalise wordVector/conceptTime for hidden nodes
		self.DPsubgraphSize = 1	#for dependencyParser	#used to normalise wordVector/conceptTime for subgraphs
		self.conceptWordVector = conceptWordVector	#requires /SPsubgraphSize
		self.conceptTime = conceptTime	#requires /SPsubgraphSize
		self.w = w #temporary sentence word index (used for reference resolution only)
		self.CPwMin = CPwMin	#temporary sentence word index (used for reference resolution only) - min of all hidden nodes
		self.CPwMax = CPwMax	#temporary sentence word index (used for reference resolution only) - max of all hidden nodes
		self.DPwMin = w	#temporary sentence word index (used for reference resolution only) - min of all hidden nodes
		self.DPwMax = w	#temporary sentence word index (used for reference resolution only) - max of all hidden nodes
		self.CPtreeLevel = CPtreeLevel	#for constituencyParser
		self.DPtreeLevel = 0	#for dependencyParser
		self.sentenceIndex = sentenceIndex
		#self.referenceSentence = False	#temporary flag: node has been reference by current sentence (used for reference resolution only)
		self.CPisPrimarySourceNode = False #temporary for SPNLPpy_syntacticalGraphConstituencyParserWordVectors only
		self.CPprimaryLeafNode = None	#temporary for SPNLPpy_syntacticalGraphConstituencyParserWordVectors only
		self.DPdependencyRelationLabelList = []	#not used (stored for reference)	#stored in dependents (to governor)	#should only contain one element
		
		#connection vars;
		self.CPgraphNodeTargetList = []	#for constituencyParser	#should only contain one element
		self.CPgraphNodeSourceList = []	#for constituencyParse
		#self.CPgraphNodeTargetDict = {}	#dict indexed by lemma, every entry is a dictionary of SyntacticalNode instances indexed by instanceID 	#for optimised lookup by concept
		#self.CPgraphNodeSourceDict = {}	#dict indexed by lemma, every entry is a dictionary of SyntacticalNode instances indexed by instanceID	#for optimised lookup by concept
		#self.foundRecentIndex = False	#temporary var (indicates referencing a previously declared instance in the article)
		self.CPsourceNodePosition = sourceNodePositionUnknown	#for leaf nodes only 
		self.DPgovernorList = []	#for dependencyParser	#should only contain one element
		self.DPdependentList = []	#for dependencyParser
		self.AGconnectionList = []	#for acyclic graph
		
		#intermediary vars for semantic graph generation;
		self.entityType = -1	#temp #GIA_ENTITY_TYPE_UNDEFINED
		self.CPmultiwordLeafNode = False
		self.referenceSetDelimiter = False	#if entityType IsRelationship only
		self.subreferenceSetDelimiter = False	#if entityType IsRelationship only
		
		#temporary graph draw variables
		self.drawn = False
		
		#temporary SPNLPpy_syntacticalGraphDependencyParserWordVectorsAcyclic variables
		self.AGtraced = False
		self.AGtracedFirst = False
		
def addConnectionToNodeTargets(node, nodeToConnect):
	node.CPgraphNodeTargetList.append(nodeToConnect)
	#addInstanceNodeToDictionary(node.CPgraphNodeTargetDict, nodeToConnect.lemma, nodeToConnect.instanceID, nodeToConnect)

def addConnectionToNodeSources(node, nodeToConnect):
	node.CPgraphNodeSourceList.append(nodeToConnect)
	#addInstanceNodeToDictionary(node.CPgraphNodeSourceDict, nodeToConnect.lemma, nodeToConnect.instanceID, nodeToConnect)
		
def removeNodeConnections(node1):	
	removeNodeSourceConnections(node1)
	removeNodeTargetConnections(node1)

def removeNodeSourceConnections(node1):	
	for node1sourceIndex, node1source in enumerate(node1.CPgraphNodeSourceList):
		for node1sourceTargetIndex, node1sourceTarget in enumerate(node1source.CPgraphNodeTargetList):
			if(node1sourceTarget == node1):
				node1sourceTargetIndexDel = node1sourceTargetIndex
		del node1source.CPgraphNodeTargetList[node1sourceTargetIndexDel]
	node1.CPgraphNodeSourceList.clear()

def removeNodeTargetConnections(node1):	
	for node1targetIndex, node1target in enumerate(node1.CPgraphNodeTargetList):
		for node1targetSourceIndex, node1targetSource in enumerate(node1target.CPgraphNodeSourceList):
			if(node1targetSource == node1):
				node1targetSourceIndexDel = node1targetSourceIndex		
		del node1target.CPgraphNodeSourceList[node1targetSourceIndexDel]
	node1.CPgraphNodeTargetList.clear()
