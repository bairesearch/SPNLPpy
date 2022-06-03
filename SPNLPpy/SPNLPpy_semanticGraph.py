"""SPNLPpy_semanticGraph.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SPNLPpy_main.py

# Usage:
see SPNLPpy_main.py

# Description:
SPNLP Semantic Graph - generate semantic graph/network based on a transformation of the SPNLP syntactical tree/graph

- semantic graph constructed based on GIA specification/implementation (SPNLPpy_semanticGraph is a minimal python implementation of GIA)
- GIAposRelTranslator applies a similar (axis orthogonal) transformation to a SANI graph to generate a GIA semantic graph (semantic relation/connection identification)

"""

import numpy as np
import spacy
from SPNLPpy_semanticNodeClass import *
from SPNLPpy_syntacticalNodeClass import *

supportMultiwordVerbsPrepositions = True

drawSemanticGraphSentence = False
if(drawSemanticGraphSentence):
	import SPNLPpy_semanticGraphDraw as SPNLPpy_semanticGraphDrawSentence
drawSemanticGraphNetwork = False
if(drawSemanticGraphNetwork):
	import SPNLPpy_semanticGraphDraw as SPNLPpy_semanticGraphDrawNetwork

performReferenceResolution = True

semanticGraphNodeDictionary = {}

#def initialiseSemanticGraph():
#	pass
	
#def finaliseSemanticGraph():
#	if(drawSemanticGraphNetwork):
#		SPNLPpy_semanticGraphDrawNetwork.displaySemanticGraph()
		
def generateSemanticGraphNetwork(articles):

	for sentenceIndex, sentence in enumerate(articles):
		generateSyntacticalGraphSentenceString(sentenceIndex, sentence)		

	if(drawSyntacticalGraphNetwork):
		SPNLPpy_syntacticalGraphDrawNetwork.displaySyntacticalGraph()
		
						
def generateSemanticGraphSentence(sentenceSyntacticalLeafNodeList, sentenceSyntacticalTreeNodeList, syntacticalGraphHeadNode, generateSemanticGraphNetwork):

	if(drawSemanticGraphSentence):
		SPNLPpy_semanticGraphDrawSentence.clearSemanticGraph()
	if(generateSemanticGraphNetwork):
		if(drawSemanticGraphNetwork):
			SPNLPpy_semanticGraphDrawNetwork.clearSemanticGraph()	
			
	sentenceSemanticNodeList = []
	generateSemanticNodes(sentenceSemanticNodeList, sentenceSyntacticalLeafNodeList)		
	#convertToSemanticGraph(sentenceSemanticNodeList, sentenceSyntacticalLeafNodeList, sentenceSyntacticalTreeNodeList, syntacticalGraphHeadNode)	#TODO

	if(drawSemanticGraphSentence):
		SPNLPpy_semanticGraphDrawSentence.displaySemanticGraph()
	
	if(generateSemanticGraphNetwork):
		if(performReferenceResolution):
			#peform reference resolution after building syntactical tree (any instance of successful reference identification will insert syntactical tree into syntactical graph/network)
			identifyBranchReferences(sentenceSemanticNodeList, graphSemanticNodeList)

		if(drawSemanticGraphNetwork):
			SPNLPpy_semanticGraphDrawNetwork.drawSemanticGraphNetwork(headNodeList)
			print("SPNLPpy_semanticGraphDrawNetwork.displaySyntacticalGraph()")
			SPNLPpy_semanticGraphDrawNetwork.displaySemanticGraph()
			
def generateSemanticNodes(sentenceSemanticNodeList, sentenceSyntacticalLeafNodeList):
	for syntacticalNode in sentenceSyntacticalLeafNodeList:
		#assume syntacticalNode.entityType already generated via performIntermediarySyntacticalTransformation
		entityName = syntacticalNode.lemma 
		semanticNode = SemanticNode(syntacticalNode.instanceID, entityName, syntacticalNode.word, syntacticalNode.wordVector, syntacticalNode.entityType)
		sentenceSemanticNodeList.append(semanticNode)
		if(drawSemanticGraph):
			SPNLPpy_semanticGraphDraw.drawSemanticGraphNode(entityName)	#do not manually define a position (automatically generated)


