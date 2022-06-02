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
import SPNLPpy_semanticGraphIntermediaryTransformation

supportMultiwordVerbsPrepositions = True

drawSemanticGraphSentence = False
if(drawSemanticGraphSentence):
	import SPNLPpy_semanticGraphDraw as SPNLPpy_semanticGraphDrawSentence
drawSemanticGraphNetwork = False
if(drawSemanticGraphNetwork):
	import SPNLPpy_semanticGraphDraw as SPNLPpy_semanticGraphDrawNetwork


def initialiseSemanticGraph():
	pass
	
def finaliseSemanticGraph():
	if(drawSemanticGraphNetwork):
		SPNLPpy_semanticGraphDrawNetwork.displaySemanticGraph()
		
def generateSemanticGraphNetwork(articles, performIntermediarySemanticTransformation):

	for sentenceIndex, sentence in enumerate(articles):
		generateSyntacticalGraphSentenceString(sentenceIndex, sentence, performIntermediarySemanticTransformation)		

	if(drawSyntacticalGraphNetwork):
		SPNLPpy_syntacticalGraphDrawNetwork.displaySyntacticalGraph()
		
						
def generateSemanticGraph(sentenceSyntacticalLeafNodeList, sentenceSyntacticalTreeNodeList, syntacticalGraphHeadNode, performIntermediarySemanticTransformation):

	if(performIntermediarySemanticTransformation):
		SPNLPpy_semanticGraphIntermediaryTransformation.performIntermediarySemanticTransformation(sentenceSyntacticalLeafNodeList, sentenceSyntacticalTreeNodeList, syntacticalGraphHeadNode)

	if(drawSemanticGraphSentence):
		SPNLPpy_semanticGraphDraw.clearSemanticGraph()
	
	sentenceSemanticNodeList = []
	generateSemanticNodes(sentenceSemanticNodeList, sentenceSyntacticalLeafNodeList)		
	#convertToSemanticGraph(sentenceSemanticNodeList, sentenceSyntacticalLeafNodeList, sentenceSyntacticalTreeNodeList)	#TODO

	if(drawSemanticGraphSentence):
		SPNLPpy_semanticGraphDraw.displaySemanticGraph()


def generateSemanticNodes(sentenceSemanticNodeList, sentenceSyntacticalLeafNodeList):
	for syntacticalNode in sentenceSyntacticalLeafNodeList:
		#assume syntacticalNode.entityType already generated via performIntermediarySemanticTransformation
		entityName = syntacticalNode.lemma 
		semanticNode = SemanticNode(syntacticalNode.instanceID, entityName, syntacticalNode.word, syntacticalNode.wordVector, syntacticalNode.entityType)
		sentenceSemanticNodeList.append(semanticNode)
		if(drawSemanticGraph):
			SPNLPpy_semanticGraphDraw.drawSemanticGraphNode(entityName)	#do not manually define a position (automatically generated)


