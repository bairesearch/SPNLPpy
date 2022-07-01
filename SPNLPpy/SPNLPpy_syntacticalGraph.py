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

- SPNLP syntactical graph (SPNLPpy_syntacticalGraphConstituencyParserWordVectors) is not equivalent to a formal/strict syntax tree or SANI tree, so apply a custom intermediary transformation

"""

import numpy as np
import spacy
spacyWordVectorGenerator = spacy.load('en_core_web_md')	#spacy.load('en_core_web_lg')
from SPNLPpy_syntacticalNodeClass import *
import SPNLPpy_syntacticalGraphOperations
import SPNLPpy_syntacticalGraphIntermediaryTransformation

useSPNLPcustomSyntacticalParser = False

if(useSPNLPcustomSyntacticalParser):
	constituencyParserType = "constituencyParserWordVector"	#default algorithmSPNLP:generateSyntacticalGraph	#experimental
else:
	constituencyParserType = "constituencyParserFormal"
if(constituencyParserType == "constituencyParserWordVector"):
	import SPNLPpy_syntacticalGraphConstituencyParserWordVectors
elif(constituencyParserType == "constituencyParserFormal"):
	import SPNLPpy_syntacticalGraphConstituencyParserFormal	
	SPNLPpy_syntacticalGraphConstituencyParserFormal.initalise(spacyWordVectorGenerator)

if(useSPNLPcustomSyntacticalParser):
	dependencyParserType = "dependencyParserWordVector"	#default algorithmSPNLP:generateSyntacticalGraph	#experimental
else:
	dependencyParserType = "dependencyParserFormal"
if(dependencyParserType == "dependencyParserWordVector"):
	import SPNLPpy_syntacticalGraphDependencyParserWordVectors
elif(dependencyParserType == "dependencyParserFormal"):
	import SPNLPpy_syntacticalGraphDependencyParserFormal
	
drawSyntacticalGraphSentence = True
if(drawSyntacticalGraphSentence):
	import SPNLPpy_syntacticalGraphDraw as SPNLPpy_syntacticalGraphDrawSentence
drawSyntacticalGraphNetwork	= True	#draw graph for entire network (not just sentence)
if(drawSyntacticalGraphNetwork):
	import SPNLPpy_syntacticalGraphDraw as SPNLPpy_syntacticalGraphDrawNetwork
drawSyntacticalGraphNodeColours = False	#enable for debugging SPNLPpy_syntacticalGraphIntermediaryTransformation
if(drawSyntacticalGraphNodeColours):
	from SPNLPpy_semanticNodeClass import identifyEntityType

performReferenceResolution = True


syntacticalGraphNodeDictionary = {}	#dict indexed by lemma, every entry is a dictionary of SyntacticalNode instances indexed by instanceID (first instance is special; reserved for concept)
#syntacticalGraphConnectionsDictionary = {}	#dict indexed tuples (lemma1, instanceID1, lemma2, instanceID2), every entry is a tuple of SyntacticalNode instances/concepts (instanceNode1, instanceNode2) [directionality: 1=source, 2=target]
	#this is used for visualisation/fast lookup purposes only - can trace node CPgraphNodeTargetList/CPgraphNodeSourceList instead

networkHeadNodeList = []

def generateSyntacticalGraphNetwork(articles, performIntermediarySyntacticalTransformation, generateSyntacticalGraphNetwork, identifySyntacticalDependencyRelations):
		
	for sentenceIndex, sentence in enumerate(articles):
		generateSyntacticalGraphSentenceString(sentenceIndex, sentence, performIntermediarySyntacticalTransformation, generateSyntacticalGraphNetwork, identifySyntacticalDependencyRelations)		
	
	return syntacticalGraphNodeDictionary	#, syntacticalGraphConnectionsDictionary
	
def generateSyntacticalGraphSentenceString(sentenceIndex, sentence, performIntermediarySyntacticalTransformation, generateSyntacticalGraphNetwork, identifySyntacticalDependencyRelations):

	print("\n\ngenerateSyntacticalGraphSentenceString: sentenceIndex = ", sentenceIndex, "; ", sentence)

	tokenisedSentence = tokeniseSentence(sentence)
	sentenceLength = len(tokenisedSentence)
	print("sentenceLength = ", sentenceLength)
	
	if(sentenceLength > 1):
		return generateSyntacticalGraphSentence(sentenceIndex, tokenisedSentence, performIntermediarySyntacticalTransformation, generateSyntacticalGraphNetwork, identifySyntacticalDependencyRelations)
	else:
		print("generateSyntacticalGraphSentenceString error: sentenceLength !> 1")
		#exit()
			
def generateSyntacticalGraphSentence(sentenceIndex, tokenisedSentence, performIntermediarySyntacticalTransformation, generateSyntacticalGraphNetwork, identifySyntacticalDependencyRelations):

	SPNLPpy_syntacticalGraphDrawSentence.setColourSyntacticalNodes(drawSyntacticalGraphNodeColours)
	#print("SPNLPpy_syntacticalGraph: SPNLPpy_syntacticalGraphDrawSentence.drawSyntacticalGraphNodeColours = ", SPNLPpy_syntacticalGraphDrawSentence.drawSyntacticalGraphNodeColours)
	
	currentTime = SPNLPpy_syntacticalGraphOperations.calculateActivationTime(sentenceIndex)

	if(generateSyntacticalGraphNetwork):
		if(drawSyntacticalGraphNetwork):
			SPNLPpy_syntacticalGraphDrawNetwork.clearSyntacticalGraph()
			
	sentenceLeafNodeList = []	#local/temporary list of sentence instance nodes (before reference resolution)		
	CPsentenceTreeNodeList = []	#local/temporary list of sentence instance nodes (before reference resolution)
	CPconnectivityStackNodeList = []	#temporary list of nodes on connectivity stack
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
		CPsubgraphSize = 1
		conceptWordVector = wordVector
		conceptTime = SPNLPpy_syntacticalGraphOperations.calculateConceptTimeLeafNode(syntacticalGraphNodeDictionary, sentenceLeafNodeList, lemma, currentTime)	#units: min time diff (not recency metric)
		CPtreeLevel = 0

		#if(addSyntacticalConceptNodesToGraph):
		#	#add concept to dictionary (if non-existent) - not currently used;
		#	if lemma not in syntacticalGraphNodeDictionary:
		#		instanceID = conceptID
		#		conceptNode = SyntacticalNode(instanceID, word, lemma, wordVector, posTag, nodeGraphType, currentTime, CPsubgraphSize, conceptWordVector, conceptTime, w, w, w, CPtreeLevel, sentenceIndex)
		#		SPNLPpy_syntacticalGraphOperations.addInstanceNodeToGraph(syntacticalGraphNodeDictionary, lemma, instanceID, conceptNode)

		#add instance to local/temporary sentenceLeafNodeList (reference resolution is required before adding nodes to graph);
		instanceID = SPNLPpy_syntacticalGraphOperations.getNewInstanceID(syntacticalGraphNodeDictionary, lemma)	#same instance id will be assigned to identical lemmas in sentence (which is not approprate in the case they refer to independent instances) - will be reassign instance id after reference resolution
		instanceNode = SyntacticalNode(instanceID, word, lemma, wordVector, posTag, nodeGraphType, currentTime, CPsubgraphSize, conceptWordVector, conceptTime, w, w, w, CPtreeLevel, sentenceIndex)
		SPNLPpy_syntacticalGraphOperations.addInstanceNodeToGraph(syntacticalGraphNodeDictionary, lemma, instanceID, instanceNode)
		if(SPNLPpy_syntacticalGraphOperations.printVerbose):
			print("create new instanceNode; ", instanceNode.lemma, ": instanceID=", instanceNode.instanceID)

		#connection vars;
		sentenceLeafNodeList.append(instanceNode)
		CPsentenceTreeNodeList.append(instanceNode)
		CPconnectivityStackNodeList.append(instanceNode)

		if(drawSyntacticalGraphNodeColours):	
			entityType = identifyEntityType(instanceNode)
			instanceNode.entityType = entityType

		#if(drawSyntacticalGraphSentence):
		#   SPNLPpy_syntacticalGraphDrawSentence.drawSyntacticalGraphNode(instanceNode, w, CPtreeLevel)   #drawn later
			
	if(constituencyParserType == "constituencyParserWordVector"):
		CPgraphHeadNode = SPNLPpy_syntacticalGraphConstituencyParserWordVectors.generateSyntacticalTreeConstituencyParserWordVectors(sentenceIndex, sentenceLeafNodeList, CPsentenceTreeNodeList, CPconnectivityStackNodeList, syntacticalGraphNodeDictionary)		
	elif(constituencyParserType == "constituencyParserFormal"):
		CPgraphHeadNode = SPNLPpy_syntacticalGraphConstituencyParserFormal.generateSyntacticalTreeConstituencyParserFormal(sentenceIndex, tokenisedSentence, sentenceLeafNodeList, CPsentenceTreeNodeList, syntacticalGraphNodeDictionary)

	if(drawSyntacticalGraphSentence):
		SPNLPpy_syntacticalGraphDrawSentence.clearSyntacticalGraph()
		SPNLPpy_syntacticalGraphDrawSentence.drawSyntacticalGraphSentence(CPgraphHeadNode, SPNLPpy_syntacticalGraphDrawSentence.syntacticalGraphTypeConstituencyTree, drawGraphNetwork=False)
		print("SPNLPpy_syntacticalGraphDrawSentence.displaySyntacticalGraph(syntacticalGraphTypeConstituencyTree)")
		SPNLPpy_syntacticalGraphDrawSentence.displaySyntacticalGraph()
				
	if(performIntermediarySyntacticalTransformation):
		SPNLPpy_syntacticalGraphIntermediaryTransformation.performIntermediarySyntacticalTransformation(constituencyParserType, sentenceLeafNodeList, CPsentenceTreeNodeList, CPgraphHeadNode)
	
	if(identifySyntacticalDependencyRelations):
		if(dependencyParserType == "dependencyParserWordVector"):
			DPgraphHeadNode = SPNLPpy_syntacticalGraphDependencyParserWordVectors.generateSyntacticalTreeDependencyParserWordVectors(sentenceIndex, sentenceLeafNodeList, CPsentenceTreeNodeList, syntacticalGraphNodeDictionary, CPgraphHeadNode, performIntermediarySyntacticalTransformation)		
		elif(dependencyParserType == "dependencyParserFormal"):
			DPgraphHeadNode = SPNLPpy_syntacticalGraphDependencyParserFormal.generateSyntacticalTreeDependencyParserFormal(sentenceIndex, tokenisedSentence, sentenceLeafNodeList, CPsentenceTreeNodeList, syntacticalGraphNodeDictionary)

		if(drawSyntacticalGraphSentence):
			SPNLPpy_syntacticalGraphDrawSentence.clearSyntacticalGraph()
			SPNLPpy_syntacticalGraphDrawSentence.drawSyntacticalGraphSentence(DPgraphHeadNode, SPNLPpy_syntacticalGraphDrawSentence.syntacticalGraphTypeDependencyTree, drawGraphNetwork=False)
			print("SPNLPpy_syntacticalGraphDrawSentence.displaySyntacticalGraph(syntacticalGraphTypeDependencyTree)")
			SPNLPpy_syntacticalGraphDrawSentence.displaySyntacticalGraph()
			
		syntacticalGraphType = SPNLPpy_syntacticalGraphDrawSentence.syntacticalGraphTypeDependencyTree
		graphHeadNode = DPgraphHeadNode
	else:
		syntacticalGraphType = SPNLPpy_syntacticalGraphDrawSentence.syntacticalGraphTypeConstituencyTree
		graphHeadNode = CPgraphHeadNode
		
	networkHeadNodeList.append(graphHeadNode)
				
	if(generateSyntacticalGraphNetwork):
		if(performReferenceResolution):
			#peform reference resolution after building syntactical tree (any instance of successful reference identification will insert syntactical tree into syntactical graph/network)
			SPNLPpy_syntacticalGraphOperations.identifyBranchReferences(syntacticalGraphNodeDictionary, CPsentenceTreeNodeList, CPgraphHeadNode, currentTime)

		if(drawSyntacticalGraphNetwork):
			SPNLPpy_syntacticalGraphDrawNetwork.drawSyntacticalGraphNetwork(networkHeadNodeList, syntacticalGraphType)
			print("SPNLPpy_syntacticalGraphDrawNetwork.displaySyntacticalGraph()")
			SPNLPpy_syntacticalGraphDrawNetwork.displaySyntacticalGraph()
				
	return sentenceLeafNodeList, CPsentenceTreeNodeList, graphHeadNode


	
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

