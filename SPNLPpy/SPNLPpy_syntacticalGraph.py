"""SPNLPpy_syntacticalGraph.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SPNLPpy_main.py

# Usage:
see SPNLPpy_main.py

# Description:
SPNLP Syntactical Graph - generate syntactical tree/graph

SPNLP (or SANI) syntactical tree stucture is generated in a format similar to a constituency-based parse tree

- SPNLP syntactical graph (SPNLPpy_syntacticalGraphConstituencyParserWordVectors) is not equivalent to a formal/strict syntax tree or SANI tree, so apply a custom intermediary transformation for semantic graph construction 

"""

import numpy as np
import spacy
spacyWordVectorGenerator = spacy.load('en_core_web_md')	#spacy.load('en_core_web_lg')
from SPNLPpy_syntacticalNodeClass import *
import SPNLPpy_syntacticalGraphOperations
import SPNLPpy_semanticGraphIntermediaryTransformation

#parserType = "constituencyParserWordVector"	#default algorithmSPNLP:generateSyntacticalGraph
parserType = "constituencyParserFormal"
if(parserType == "constituencyParserWordVector"):
	import SPNLPpy_syntacticalGraphConstituencyParserWordVectors
elif(parserType == "constituencyParserFormal"):
	import SPNLPpy_syntacticalGraphConstituencyParserFormal	
	SPNLPpy_syntacticalGraphConstituencyParserFormal.initalise(spacyWordVectorGenerator)	

drawSyntacticalGraphSentence = True
if(drawSyntacticalGraphSentence):
	import SPNLPpy_syntacticalGraphDraw as SPNLPpy_syntacticalGraphDrawSentence
drawSyntacticalGraphNetwork	= True	#draw graph for entire network (not just sentence)
if(drawSyntacticalGraphNetwork):
	import SPNLPpy_syntacticalGraphDraw as SPNLPpy_syntacticalGraphDrawNetwork
drawSyntacticalGraphNodeColours = False	#enable for debugging SPNLPpy_semanticGraphIntermediaryTransformation
if(drawSyntacticalGraphNodeColours):
	from SPNLPpy_semanticNodeClass import identifyEntityType

performReferenceResolution = True


graphNodeDictionary = {}	#dict indexed by lemma, every entry is a dictionary of SyntacticalNode instances indexed by instanceID (first instance is special; reserved for concept)
#graphConnectionsDictionary = {}	#dict indexed tuples (lemma1, instanceID1, lemma2, instanceID2), every entry is a tuple of SyntacticalNode instances/concepts (instanceNode1, instanceNode2) [directionality: 1=source, 2=target]
	#this is used for visualisation/fast lookup purposes only - can trace node graphNodeTargetList/graphNodeSourceList instead

headNodeList = []

def generateSyntacticalGraphNetwork(articles, performIntermediarySemanticTransformation, generateSyntacticalGraphNetwork):
		
	for sentenceIndex, sentence in enumerate(articles):
		generateSyntacticalGraphSentenceString(sentenceIndex, sentence, performIntermediarySemanticTransformation, generateSyntacticalGraphNetwork)		
	
	return graphNodeDictionary	#, graphConnectionsDictionary
	
def generateSyntacticalGraphSentenceString(sentenceIndex, sentence, performIntermediarySemanticTransformation, generateSyntacticalGraphNetwork):

	print("\n\ngenerateSyntacticalGraphSentenceString: sentenceIndex = ", sentenceIndex, "; ", sentence)

	tokenisedSentence = tokeniseSentence(sentence)
	sentenceLength = len(tokenisedSentence)
	print("sentenceLength = ", sentenceLength)
	
	if(sentenceLength > 1):
		return generateSyntacticalGraphSentence(sentenceIndex, tokenisedSentence, performIntermediarySemanticTransformation, generateSyntacticalGraphNetwork)
	else:
		print("generateSyntacticalGraphSentenceString error: sentenceLength !> 1")
		#exit()
			
def generateSyntacticalGraphSentence(sentenceIndex, tokenisedSentence, performIntermediarySemanticTransformation, generateSyntacticalGraphNetwork):

	SPNLPpy_syntacticalGraphDrawSentence.setColourSyntacticalNodes(drawSyntacticalGraphNodeColours)
	print("SPNLPpy_syntacticalGraph: SPNLPpy_syntacticalGraphDrawSentence.drawSyntacticalGraphNodeColours = ", SPNLPpy_syntacticalGraphDrawSentence.drawSyntacticalGraphNodeColours)
	
	currentTime = SPNLPpy_syntacticalGraphOperations.calculateActivationTime(sentenceIndex)

	if(drawSyntacticalGraphSentence):
		SPNLPpy_syntacticalGraphDrawSentence.clearSyntacticalGraph()
	if(generateSyntacticalGraphNetwork):
		if(drawSyntacticalGraphNetwork):
			SPNLPpy_syntacticalGraphDrawNetwork.clearSyntacticalGraph()
			
	sentenceLeafNodeList = []	#local/temporary list of sentence instance nodes (before reference resolution)		
	sentenceTreeNodeList = []	#local/temporary list of sentence instance nodes (before reference resolution)
	connectivityStackNodeList = []	#temporary list of nodes on connectivity stack
	#sentenceGraphNodeDictionary = {}	#local/isolated/temporary graph of sentence instance nodes (before reference resolution)
		
	sentenceLength = len(tokenisedSentence)
	
	#declare graph nodes;
	for w, token in enumerate(tokenisedSentence):	

		#primary vars;
		word = getTokenWord(token)
		lemma = getTokenLemma(token)
		wordVector = getTokenWordVector(token)	#numpy word vector
		posTag = getTokenPOStag(token)
		activationTime = SPNLPpy_syntacticalGraphOperations.calculateActivationTime(sentenceIndex)
		nodeGraphType = graphNodeTypeLeaf
		
		#sentenceTreeArtificial vars;
		subgraphSize = 1
		conceptWordVector = wordVector
		conceptTime = SPNLPpy_syntacticalGraphOperations.calculateConceptTimeLeafNode(graphNodeDictionary, sentenceLeafNodeList, lemma, currentTime)	#units: min time diff (not recency metric)
		treeLevel = 0

		#if(addSyntacticalConceptNodesToGraph):
		#	#add concept to dictionary (if non-existent) - not currently used;
		#	if lemma not in graphNodeDictionary:
		#		instanceID = conceptID
		#		conceptNode = SyntacticalNode(instanceID, word, lemma, wordVector, posTag, nodeGraphType, currentTime, subgraphSize, conceptWordVector, conceptTime, w, w, w, treeLevel, sentenceIndex)
		#		SPNLPpy_syntacticalGraphOperations.addInstanceNodeToGraph(graphNodeDictionary, lemma, instanceID, conceptNode)

		#add instance to local/temporary sentenceLeafNodeList (reference resolution is required before adding nodes to graph);
		instanceID = SPNLPpy_syntacticalGraphOperations.getNewInstanceID(graphNodeDictionary, lemma)	#same instance id will be assigned to identical lemmas in sentence (which is not approprate in the case they refer to independent instances) - will be reassign instance id after reference resolution
		instanceNode = SyntacticalNode(instanceID, word, lemma, wordVector, posTag, nodeGraphType, currentTime, subgraphSize, conceptWordVector, conceptTime, w, w, w, treeLevel, sentenceIndex)
		SPNLPpy_syntacticalGraphOperations.addInstanceNodeToGraph(graphNodeDictionary, lemma, instanceID, instanceNode)
		if(SPNLPpy_syntacticalGraphOperations.printVerbose):
			print("create new instanceNode; ", instanceNode.lemma, ": instanceID=", instanceNode.instanceID)

		#connection vars;
		sentenceLeafNodeList.append(instanceNode)
		sentenceTreeNodeList.append(instanceNode)
		connectivityStackNodeList.append(instanceNode)

		if(drawSyntacticalGraphNodeColours):	
			entityType = identifyEntityType(instanceNode)
			instanceNode.entityType = entityType
		if(drawSyntacticalGraphSentence):
			SPNLPpy_syntacticalGraphDrawSentence.drawSyntacticalGraphNode(instanceNode, w, treeLevel)


	if(parserType == "constituencyParserWordVector"):
		graphHeadNode = SPNLPpy_syntacticalGraphConstituencyParserWordVectors.generateSyntacticalTreeConstituencyParserWordVectors(sentenceIndex, sentenceLeafNodeList, sentenceTreeNodeList, connectivityStackNodeList, graphNodeDictionary)		
	elif(parserType == "constituencyParserFormal"):
		graphHeadNode = SPNLPpy_syntacticalGraphConstituencyParserFormal.generateSyntacticalTreeConstituencyParserFormal(sentenceIndex, tokenisedSentence, sentenceLeafNodeList, sentenceTreeNodeList, graphNodeDictionary)

	headNodeList.append(graphHeadNode)

	if(drawSyntacticalGraphSentence):
		for hiddenNode in sentenceTreeNodeList:
			if((hiddenNode.graphNodeType == graphNodeTypeBranch) or (hiddenNode.graphNodeType == graphNodeTypeHead)):
				SPNLPpy_syntacticalGraphDrawSentence.drawSyntacticalGraphNodeAndConnections(hiddenNode, drawGraph=False)
		print("SPNLPpy_syntacticalGraphDrawSentence.displaySyntacticalGraph()")
		SPNLPpy_syntacticalGraphDrawSentence.displaySyntacticalGraph()
		
	if(performIntermediarySemanticTransformation):
		SPNLPpy_semanticGraphIntermediaryTransformation.performIntermediarySemanticTransformation(parserType, sentenceLeafNodeList, sentenceTreeNodeList, graphHeadNode)
			
	if(generateSyntacticalGraphNetwork):
		if(performReferenceResolution):
			#peform reference resolution after building syntactical tree (any instance of successful reference identification will insert syntactical tree into syntactical graph/network)
			SPNLPpy_syntacticalGraphOperations.identifyBranchReferences(graphNodeDictionary, sentenceTreeNodeList, graphHeadNode, currentTime)

		if(drawSyntacticalGraphNetwork):
			SPNLPpy_syntacticalGraphDrawNetwork.drawSyntacticalGraphNetwork(headNodeList)
			print("SPNLPpy_syntacticalGraphDrawNetwork.displaySyntacticalGraph()")
			SPNLPpy_syntacticalGraphDrawNetwork.displaySyntacticalGraph()
				
	return sentenceLeafNodeList, sentenceTreeNodeList, graphHeadNode


	
#tokenisation:

def tokeniseSentence(sentence):
	tokenList = spacyWordVectorGenerator(sentence)
	return tokenList

def getTokenWord(token):
	word = token.text
	return word
	
def getTokenLemma(token):
	lemma = token.lemma_
	if(token.lemma_ == '-PRON-'):
		lemma = token.text	#https://stackoverflow.com/questions/56966754/how-can-i-make-spacy-not-produce-the-pron-lemma
	return lemma
		
def getTokenWordVector(token):
	wordVector = token.vector	#cpu: type numpy
	return wordVector

def getTokenPOStag(token):
	#nlp in context prediction only (not certain)
	posTag = token.pos_
	return posTag

