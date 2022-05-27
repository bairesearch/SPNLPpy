"""ATNLPtf_syntacticalGraphDraw.py

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
import ATNLPtf_syntacticalNodeClass	#required for drawSyntacticalGraph only
from ATNLPtf_semanticGraphDraw import getEntityNodeColour

syntacticalGraph = nx.Graph()
syntacticalGraphNodeColorMap = []
drawSyntacticalGraphNodeColours = False

def setColourSyntacticalNodes(value):
    global drawSyntacticalGraphNodeColours
    drawSyntacticalGraphNodeColours = value

def clearSyntacticalGraph():
	syntacticalGraph.clear()	#only draw graph for single sentence
	if(drawSyntacticalGraphNodeColours):
		syntacticalGraphNodeColorMap.clear()

def drawSyntacticalGraphNode(node, w, treeLevel):
	colorHtml = getEntityNodeColour(node.entityType)
	#print("colorHtml = ", colorHtml)
	syntacticalGraph.add_node(generateSyntacticalGraphNodeName(node), pos=(w, treeLevel))	# color=colorHtml
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

def drawSyntacticalGraph(syntacticalGraphNode):	
	#parse graph and generate nodes and connections
	drawSyntacticalGraphNode(syntacticalGraphNode, syntacticalGraphNode.w, syntacticalGraphNode.treeLevel)
	for sourceNode in syntacticalGraphNode.graphNodeSourceList:
		drawSyntacticalGraphConnection(syntacticalGraphNode, sourceNode)
		drawSyntacticalGraph(sourceNode)
	
def generateSyntacticalGraphNodeName(node):
	if(drawSyntacticalGraphNodeColours):
		if(node.graphNodeType == ATNLPtf_syntacticalNodeClass.graphNodeTypeLeaf):
			#this is required to differentiate duplicate words in the sentence
			nodeName = node.lemma + str(node.w)
		else:
			#this is required in the event there are more than 2 nodes in the graph hierarchy of the same name
			nodeName = node.lemma + str(node.w)	
	else:
		nodeName = node.lemma
	return nodeName
