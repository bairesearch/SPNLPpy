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
SPNLP Syntactical Graph Constituency Parser (CP) Formal - external python constituency-based parse tree generation

Preconditions: assumes leaf nodes already generated

"""

import numpy as np
import spacy
import benepar
from SPNLPpy_syntacticalNodeClass import *
import SPNLPpy_syntacticalGraphOperations
benepar.download('benepar_en3')	#Berkeley constituency parser
 

def initalise(spacyWordVectorGenerator):
	if spacy.__version__.startswith('2'):
		spacyWordVectorGenerator.add_pipe(benepar.BeneparComponent("benepar_en3"))
	else:
		spacyWordVectorGenerator.add_pipe("benepar", config={"model": "benepar_en3"})


def generateSyntacticalTreeConstituencyParserFormal(sentenceIndex, tokenisedSentence, sentenceLeafNodeList, CPsentenceTreeNodeList, syntacticalGraphNodeDictionary):

	constituents = list(tokenisedSentence.sents)[0]
	print(constituents._.parse_string)
	graphHeadNode, wCurrentLeafNode = generateSyntacticalTree(constituents, sentenceIndex, sentenceLeafNodeList, CPsentenceTreeNodeList, syntacticalGraphNodeDictionary, True, 0)	#or constituents

	return graphHeadNode

def generateSyntacticalTree(constituent, sentenceIndex, sentenceLeafNodeList, CPsentenceTreeNodeList, syntacticalGraphNodeDictionary, isHead, wCurrentLeafNode):

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
		CPsubgraphSize = 0
		conceptWordVector = 0
		conceptTime = 0
		CPtreeLevel = 0
		wSum = 0
		CPwMin = 9999999	#initialise to very large number
		CPwMax = 0

		childNodeList = []
		for childIndex, childConstituent in enumerate(list(constituent._.children)):
			childNode, wCurrentLeafNode = generateSyntacticalTree(childConstituent, sentenceIndex, sentenceLeafNodeList, CPsentenceTreeNodeList, syntacticalGraphNodeDictionary, False, wCurrentLeafNode)
			childNodeList.append(childNode)

			#primary vars;
			word = word + childNode.word
			lemma = lemma + childNode.lemma
			wordVectorSum = childNode.wordVector + wordVectorSum
			posTag = None
			activationTime = SPNLPpy_syntacticalGraphOperations.calculateActivationTime(sentenceIndex)	#mean([connectionNode1.activationTime, connectionNode2.activationTime]) 

			#sentenceTreeArtificial vars;
			CPsubgraphSize = CPsubgraphSize + childNode.CPsubgraphSize
			conceptWordVector = np.add(childNode.conceptWordVector, conceptWordVector)
			conceptTime = conceptTime + childNode.conceptTime
			CPtreeLevel = max(CPtreeLevel, childNode.CPtreeLevel)
			wSum = wSum + childNode.w
			CPwMin = min(CPwMin, childNode.CPwMin)
			CPwMax = max(CPwMax, childNode.CPwMax)

		#print("create hidden node, lemma = ", lemma)

		#perform averages;
		CPtreeLevel = CPtreeLevel + 1
		w = wSum/numberOfChildren	#mean
		wordVector = SPNLPpy_syntacticalGraphOperations.getBranchWordVectorFromSourceNodesSum(wordVectorSum, conceptWordVector, CPsubgraphSize, numberOfChildren)
		CPsubgraphSize = CPsubgraphSize + 1
		
		instanceID = SPNLPpy_syntacticalGraphOperations.getNewInstanceID(syntacticalGraphNodeDictionary, lemma)
		hiddenNode = SyntacticalNode(instanceID, word, lemma, wordVector, posTag, nodeGraphType, currentTime, CPsubgraphSize, conceptWordVector, conceptTime, w, CPwMin, CPwMax, CPtreeLevel, sentenceIndex)
		hiddenNode.CPlabel = constituentLabel
		SPNLPpy_syntacticalGraphOperations.addInstanceNodeToGraph(syntacticalGraphNodeDictionary, lemma, instanceID, hiddenNode)
		
		#connection vars;
		for childNode in childNodeList:
			#print("hiddenNode = ", hiddenNode.lemma, ", childNode = ", childNode.lemma)
			addConnectionToNodeTargets(childNode, hiddenNode)
			addConnectionToNodeSources(hiddenNode, childNode)
			childNode.CPsourceNodePosition = childIndex
				
		CPsentenceTreeNodeList.append(hiddenNode)
	
	return hiddenNode, wCurrentLeafNode
	
