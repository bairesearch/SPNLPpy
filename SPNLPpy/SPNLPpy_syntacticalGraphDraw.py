"""SPNLPpy_syntacticalGraphDraw.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SPNLPpy_main.py

# Usage:
see SPNLPpy_main.py

# Description:
SPNLP Syntactical Graph Draw Class

"""

import networkx as nx
import matplotlib.pyplot as plt
import SPNLPpy_syntacticalNodeClass	#required for drawSyntacticalGraphSentence only
from SPNLPpy_semanticGraphDraw import getEntityNodeColour

syntacticalGraphTypeConstituencyTree = 1
syntacticalGraphTypeDependencyTree = 2

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

def drawSyntacticalGraphSentence(syntacticalGraphNode, syntacticalGraphType, drawGraphNetwork=False):	
	#parse tree and generate nodes and connections
	
	drawNode = True
	if(drawGraphNetwork):
		sentenceIndex = syntacticalGraphNode.sentenceIndex
		#print("sentenceIndex = ", sentenceIndex)
		if(syntacticalGraphNode.drawn):
			drawNode = False
		else:
			syntacticalGraphNode.drawn = True
	else:
		 sentenceIndex = 0
		 
	if(drawNode):
		if(syntacticalGraphType == syntacticalGraphTypeConstituencyTree):
			drawSyntacticalGraphNode(syntacticalGraphNode, syntacticalGraphNode.w, syntacticalGraphNode.CPtreeLevel, sentenceIndex)
			for sourceNode in syntacticalGraphNode.CPgraphNodeSourceList:
				drawSyntacticalGraphConnection(syntacticalGraphNode, sourceNode)
				drawSyntacticalGraphSentence(sourceNode, syntacticalGraphType, drawGraphNetwork)	
		elif(syntacticalGraphType == syntacticalGraphTypeDependencyTree):
			drawSyntacticalGraphNode(syntacticalGraphNode, syntacticalGraphNode.w, syntacticalGraphNode.DPtreeLevel, sentenceIndex)
			for sourceNode in syntacticalGraphNode.DPdependentList:
				drawSyntacticalGraphConnection(syntacticalGraphNode, sourceNode)
				drawSyntacticalGraphSentence(sourceNode, syntacticalGraphType, drawGraphNetwork)
					
def generateSyntacticalGraphNodeName(node):
	if(drawSyntacticalGraphNodeColours):
		if(node.graphNodeType == SPNLPpy_syntacticalNodeClass.graphNodeTypeLeaf):
			#this is required to differentiate duplicate words in the sentence/article
			nodeName = node.lemma + str("w") + str(node.w) + str("s") + str(node.sentenceIndex) 
		else:
			#this is required in the event there are more than 2 nodes in the sentence/article graph hierarchy of the same name
			nodeName = node.lemma + str("w") + str(node.w) + str("s") + str(node.sentenceIndex) 
	else:
		nodeName = node.lemma
	return nodeName

def drawSyntacticalGraphNetwork(headNodeList, syntacticalGraphType):	
	#parse graph and generate nodes and connections
	for headNode in headNodeList:
		#print("headNode.lemma = ", headNode.lemma)
		drawSyntacticalGraphSentence(headNode, syntacticalGraphType, drawGraphNetwork=True)
	for headNode in headNodeList:
		drawSyntacticalGraphSentenceReset(headNode, syntacticalGraphType)

def drawSyntacticalGraphSentenceReset(syntacticalGraphNode, syntacticalGraphType):	
	if(syntacticalGraphNode.drawn):
		syntacticalGraphNode.drawn = False
		if(syntacticalGraphType == syntacticalGraphTypeConstituencyTree):
			for sourceNode in syntacticalGraphNode.CPgraphNodeSourceList:
				drawSyntacticalGraphSentenceReset(sourceNode, syntacticalGraphType)		
		elif(syntacticalGraphType == syntacticalGraphTypeDependencyTree):
			for sourceNode in syntacticalGraphNode.DPdependentList:
				drawSyntacticalGraphSentenceReset(sourceNode, syntacticalGraphType)

def drawSyntacticalGraphNodeAndConnections(syntacticalGraphNode, syntacticalGraphType, drawGraphNetwork=False):	
	if(drawGraphNetwork):
		sentenceIndex = syntacticalGraphNode.sentenceIndex
	else:
		 sentenceIndex = 0
	if(syntacticalGraphType == syntacticalGraphTypeConstituencyTree):
		drawSyntacticalGraphNode(syntacticalGraphNode, syntacticalGraphNode.w, syntacticalGraphNode.CPtreeLevel, sentenceIndex)
		for sourceNode in syntacticalGraphNode.CPgraphNodeSourceList:
			drawSyntacticalGraphConnection(syntacticalGraphNode, sourceNode)
	elif(syntacticalGraphType == syntacticalGraphTypeDependencyTree):
		drawSyntacticalGraphNode(syntacticalGraphNode, syntacticalGraphNode.w, syntacticalGraphNode.DPtreeLevel, sentenceIndex)
		for sourceNode in syntacticalGraphNode.DPdependentList:
			drawSyntacticalGraphConnection(syntacticalGraphNode, sourceNode)
		
