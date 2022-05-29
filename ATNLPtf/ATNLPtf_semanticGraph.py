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
import ATNLPtf_semanticGraphIntermediaryTransformation

supportMultiwordVerbsPrepositions = True

drawSemanticGraphSentence = False
if(drawSemanticGraphSentence):
	import ATNLPtf_semanticGraphDraw as ATNLPtf_semanticGraphDrawSentence
drawSemanticGraphNetwork = False
if(drawSemanticGraphNetwork):
	import ATNLPtf_semanticGraphDraw as ATNLPtf_semanticGraphDrawNetwork


def initialiseSemanticGraph():
	pass
	
def finaliseSemanticGraph():
	if(drawSemanticGraphNetwork):
		ATNLPtf_semanticGraphDrawNetwork.displaySemanticGraph()
		
def generateSemanticGraphNetwork(articles, performIntermediarySemanticTransformation):

	for sentenceIndex, sentence in enumerate(articles):
		generateSyntacticalGraphSentenceString(sentenceIndex, sentence, performIntermediarySemanticTransformation)		

	if(drawSyntacticalGraphNetwork):
		ATNLPtf_syntacticalGraphDrawNetwork.displaySyntacticalGraph()
		
						
def generateSemanticGraph(sentenceSyntacticalLeafNodeList, sentenceSyntacticalTreeNodeList, syntacticalGraphHeadNode, performIntermediarySemanticTransformation):

	if(performIntermediarySemanticTransformation):
		ATNLPtf_semanticGraphIntermediaryTransformation.performIntermediarySemanticTransformation(sentenceSyntacticalLeafNodeList, sentenceSyntacticalTreeNodeList, syntacticalGraphHeadNode)

	if(drawSemanticGraphSentence):
		ATNLPtf_semanticGraphDraw.clearSemanticGraph()
	
	sentenceSemanticNodeList = []
	generateSemanticNodes(sentenceSemanticNodeList, sentenceSyntacticalLeafNodeList)		
	#convertToSemanticGraph(sentenceSemanticNodeList, sentenceSyntacticalLeafNodeList, sentenceSyntacticalTreeNodeList)	#TODO

	if(drawSemanticGraphSentence):
		ATNLPtf_semanticGraphDraw.displaySemanticGraph()


def generateSemanticNodes(sentenceSemanticNodeList, sentenceSyntacticalLeafNodeList):
	for syntacticalNode in sentenceSyntacticalLeafNodeList:
		#assume syntacticalNode.entityType already generated via performIntermediarySemanticTransformation
		entityName = syntacticalNode.lemma 
		semanticNode = SemanticNode(syntacticalNode.instanceID, entityName, syntacticalNode.word, syntacticalNode.wordVector, syntacticalNode.entityType)
		sentenceSemanticNodeList.append(semanticNode)
		if(drawSemanticGraph):
			ATNLPtf_semanticGraphDraw.drawSemanticGraphNode(entityName)	#do not manually define a position (automatically generated)


