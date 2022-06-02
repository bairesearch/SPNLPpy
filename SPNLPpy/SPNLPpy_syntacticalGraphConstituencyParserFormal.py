"""SPNLPpy_syntacticalGraphConstituencyParserFormal.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SPNLPpy_main.py

# Usage:
see SPNLPpy_main.py

# Description:
SPNLP Syntactical Graph Constituency Parser Formal - external python constituency-based parse tree generation

Preconditions: assumes leaf nodes already generated

"""

import numpy as np
import spacy
import benepar
from SPNLPpy_syntacticalNodeClass import *
import SPNLPpy_syntacticalGraphOperations
import nltk
benepar.download('benepar_en3')	#Berkeley constituency parser
 

def initalise(spacyWordVectorGenerator):
	if spacy.__version__.startswith('2'):
		spacyWordVectorGenerator.add_pipe(benepar.BeneparComponent("benepar_en3"))
	else:
		spacyWordVectorGenerator.add_pipe("benepar", config={"model": "benepar_en3"})


def generateSyntacticalTreeConstituencyParserFormal(sentenceIndex, tokenisedSentence, sentenceLeafNodeList, sentenceTreeNodeList, graphNodeDictionary):

	constituents = list(tokenisedSentence.sents)[0]
	print(constituents._.parse_string)
	graphHeadNode, wCurrentLeafNode = generateSyntacticalTree(constituents, sentenceIndex, sentenceLeafNodeList, sentenceTreeNodeList, graphNodeDictionary, True, 0)	#or constituents

	return graphHeadNode

def generateSyntacticalTree(constituent, sentenceIndex, sentenceLeafNodeList, sentenceTreeNodeList, graphNodeDictionary, isHead, wCurrentLeafNode):

	constituentText = constituent
	constituentLabel = constituent._.labels	#or constituent.labels
	if(SPNLPpy_syntacticalGraphOperations.printVerbose):
		print("generateSyntacticalTree: constituentText = ", constituentText, ", constituentLabel = ", constituentLabel)
	
	currentTime = SPNLPpy_syntacticalGraphOperations.calculateActivationTime(sentenceIndex)

	numberOfChildren = len(list(constituent._.children))
	if(isHead):
		nodeGraphType = graphNodeTypeHead
	else:
		if(numberOfChildren > 0):	#or constituent.children:
			nodeGraphType = graphNodeTypeBranch
		else:
			nodeGraphType = graphNodeTypeLeaf
	
	if(nodeGraphType == graphNodeTypeLeaf):
		hiddenNode = sentenceLeafNodeList[wCurrentLeafNode]
		wCurrentLeafNode = wCurrentLeafNode + 1	#assumes constituency parser leaf nodes are accessed/parsed in word order
	else:
		#primary vars;
		word = ""	#constituent
		lemma = ""
		wordVectorSum = 0
		posTag = None
		activationTime = SPNLPpy_syntacticalGraphOperations.calculateActivationTime(sentenceIndex)	#mean([connectionNode1.activationTime, connectionNode2.activationTime]) 

		#sentenceTreeArtificial vars;
		subgraphSize = 0
		conceptWordVector = 0
		conceptTime = 0
		treeLevel = 0
		wSum = 0
		wMin = 9999999	#initialise to very large number
		wMax = 0

		childNodeList = []
		for childIndex, childConstituent in enumerate(list(constituent._.children)):
			childNode, wCurrentLeafNode = generateSyntacticalTree(childConstituent, sentenceIndex, sentenceLeafNodeList, sentenceTreeNodeList, graphNodeDictionary, False, wCurrentLeafNode)
			childNodeList.append(childNode)

			#primary vars;
			word = word + childNode.word
			lemma = lemma + childNode.lemma
			wordVectorSum = childNode.wordVector + wordVectorSum
			posTag = None
			activationTime = SPNLPpy_syntacticalGraphOperations.calculateActivationTime(sentenceIndex)	#mean([connectionNode1.activationTime, connectionNode2.activationTime]) 

			#sentenceTreeArtificial vars;
			subgraphSize = subgraphSize + childNode.subgraphSize
			conceptWordVector = np.add(childNode.conceptWordVector, conceptWordVector)
			conceptTime = conceptTime + childNode.conceptTime
			treeLevel = max(treeLevel, childNode.treeLevel)
			wSum = wSum + childNode.w
			wMin = min(wMin, childNode.wMin)
			wMax = max(wMax, childNode.wMax)

		#print("create hidden node, lemma = ", lemma)

		#perform averages;
		treeLevel = treeLevel + 1
		w = wSum/numberOfChildren	#mean
		wordVector = SPNLPpy_syntacticalGraphOperations.getBranchWordVectorFromSourceNodesSum(wordVectorSum, conceptWordVector, subgraphSize, numberOfChildren)
		subgraphSize = subgraphSize + 1
		
		instanceID = SPNLPpy_syntacticalGraphOperations.getNewInstanceID(graphNodeDictionary, lemma)
		hiddenNode = SyntacticalNode(instanceID, word, lemma, wordVector, posTag, nodeGraphType, currentTime, subgraphSize, conceptWordVector, conceptTime, w, wMin, wMax, treeLevel, sentenceIndex)
		hiddenNode.constituencyParserLabel = constituentLabel
		SPNLPpy_syntacticalGraphOperations.addInstanceNodeToGraph(graphNodeDictionary, lemma, instanceID, hiddenNode)
		
		#connection vars;
		for childNode in childNodeList:
			#print("hiddenNode = ", hiddenNode.lemma, ", childNode = ", childNode.lemma)
			addConnectionToNodeTargets(childNode, hiddenNode)
			addConnectionToNodeSources(hiddenNode, childNode)
			childNode.sourceNodePosition = childIndex
				
		sentenceTreeNodeList.append(hiddenNode)
	
	return hiddenNode, wCurrentLeafNode
	
