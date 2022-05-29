"""ATNLPtf_syntacticalGraphDrawSentence.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATNLPtf_main.py

# Usage:
see ATNLPtf_main.py

# Description:
ATNLP Syntactical Graph Draw Class

"""

import networkx as nx
import matplotlib.pyplot as plt
import ATNLPtf_syntacticalNodeClass	#required for drawSyntacticalGraphSentence only
from ATNLPtf_semanticGraphDraw import getEntityNodeColour

syntacticalGraph = nx.Graph()
syntacticalGraphNodeColorMap = []
drawSyntacticalGraphNodeColours = False
maxWordLeafNodesDrawnPerSentence = 20

def setColourSyntacticalNodes(value):
    global drawSyntacticalGraphNodeColours
    drawSyntacticalGraphNodeColours = value

def clearSyntacticalGraph():
	syntacticalGraph.clear()	#only draw graph for single sentence
	if(drawSyntacticalGraphNodeColours):
		syntacticalGraphNodeColorMap.clear()

def drawSyntacticalGraphNode(node, w, treeLevel, sentenceIndex=0):
	colorHtml = getEntityNodeColour(node.entityType)
	#print("colorHtml = ", colorHtml)
	posX = sentenceIndex*maxWordLeafNodesDrawnPerSentence + w
	syntacticalGraph.add_node(generateSyntacticalGraphNodeName(node), pos=(posX, treeLevel))	# color=colorHtml
	if(drawSyntacticalGraphNodeColours):
		syntacticalGraphNodeColorMap.append(colorHtml)

def drawSyntacticalGraphConnection(node1, node2):
	syntacticalGraph.add_edge(generateSyntacticalGraphNodeName(node1), generateSyntacticalGraphNodeName(node2))

def displaySyntacticalGraph():
	pos = nx.get_node_attributes(syntacticalGraph, 'pos')
	if(drawSyntacticalGraphNodeColours):
		nx.draw(syntacticalGraph, pos, node_color=syntacticalGraphNodeColorMap, with_labels=True)	
	else:
		nx.draw(syntacticalGraph, pos, with_labels=True)
	plt.show()

def drawSyntacticalGraphSentence(syntacticalGraphNode, drawGraph=False):	
	#parse tree and generate nodes and connections
	drawNode = True
	if(drawGraph):
		sentenceIndex = syntacticalGraphNode.sentenceIndex
		#print("sentenceIndex = ", sentenceIndex)
		if(syntacticalGraphNode.drawn):
			drawNode = False
		else:
			syntacticalGraphNode.drawn = True
	else:
		 sentenceIndex = 0
	if(drawNode):
		drawSyntacticalGraphNode(syntacticalGraphNode, syntacticalGraphNode.w, syntacticalGraphNode.treeLevel, sentenceIndex)
		for sourceNode in syntacticalGraphNode.graphNodeSourceList:
			drawSyntacticalGraphConnection(syntacticalGraphNode, sourceNode)
			drawSyntacticalGraphSentence(sourceNode, drawGraph)
	
def generateSyntacticalGraphNodeName(node):
	if(drawSyntacticalGraphNodeColours):
		if(node.graphNodeType == ATNLPtf_syntacticalNodeClass.graphNodeTypeLeaf):
			#this is required to differentiate duplicate words in the sentence/article
			nodeName = node.lemma + str("w") + str(node.w) + str("s") + str(node.sentenceIndex) 
		else:
			#this is required in the event there are more than 2 nodes in the sentence/article graph hierarchy of the same name
			nodeName = node.lemma + str("w") + str(node.w) + str("s") + str(node.sentenceIndex) 
	else:
		nodeName = node.lemma
	return nodeName

def drawSyntacticalGraphNetwork(headNodeList):	
	#parse graph and generate nodes and connections
	for headNode in headNodeList:
		#print("headNode.lemma = ", headNode.lemma)
		drawSyntacticalGraphSentence(headNode, drawGraph=True)
	for headNode in headNodeList:
		drawSyntacticalGraphSentenceReset(headNode)

def drawSyntacticalGraphSentenceReset(syntacticalGraphNode):	
	if(syntacticalGraphNode.drawn):
		syntacticalGraphNode.drawn = False
		for sourceNode in syntacticalGraphNode.graphNodeSourceList:
			drawSyntacticalGraphSentenceReset(sourceNode)	
