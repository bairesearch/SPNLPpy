"""ATNLPtf_main.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
Python 3 and Tensorflow 2.1+ 

conda create -n anntf2 python=3.7
source activate anntf2
conda install -c tensorflow tensorflow=2.3
conda install nltk
conda install spacy
python3 -m spacy download en_core_web_md

# Usage:
python3 ATNLPtf_main.py

# Description:
ATNLP - axis transformation natural language processing

"""

# %tensorflow_version 2.x
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from nltk.tokenize import word_tokenize
from collections import Counter, defaultdict

import ATNLPtf_getAllPossiblePosTags

import tensorflow as tf
import numpy as np

import sys
np.set_printoptions(threshold=sys.maxsize)

import random
import ANNtf2_loadDataset


#ATNLP algorithm selection;
#algorithmATNLP = "normaliseInputVectors"	#original ATNLP method
algorithmATNLP = "generateGraph"	#method of syntactical/semantic graph construction based on word proximity, frequency, and recency

#debug parameters
debugUseSmallSequentialInputDataset = False

if(algorithmATNLP == "normaliseInputVectors"):
	import ATNLPtf_normalisation
	NLPsequentialInputTypeTokeniseWords = True	#perform nltk tokenization early in pipeline
elif(algorithmATNLP == "generateGraph"):
	import ATNLPtf_graph
	NLPsequentialInputTypeTokeniseWords = False	#perform spacy tokenization later in pipeline

supportBatchAndMultiAbstractionLevelProcessing = False		#batches are not currently processed/normalised in parallel (retained for source base compatibility);
if(supportBatchAndMultiAbstractionLevelProcessing):
	import ATNLPtf_processingBatchAndMultiAbstractionLevel
	NLPsequentialInputTypeMinWordVectors = ATNLPtf_processingBatchAndMultiAbstractionLevel.NLPsequentialInputTypeMinWordVectors
	NLPsequentialInputTypeMaxWordVectors = ATNLPtf_processingBatchAndMultiAbstractionLevel.NLPsequentialInputTypeMaxWordVectors
	limitSentenceLengthsSize = ATNLPtf_processingBatchAndMultiAbstractionLevel.limitSentenceLengthsSize
	limitSentenceLengths = ATNLPtf_processingBatchAndMultiAbstractionLevel.useSmallSentenceLengths
	NLPsequentialInputTypeTrainWordVectors = ATNLPtf_processingBatchAndMultiAbstractionLevel.NLPsequentialInputTypeTrainWordVectors	
else:
	#mandatory for !supportBatchAndMultiAbstractionLevelProcessing;
	NLPsequentialInputTypeMinWordVectors = True
	NLPsequentialInputTypeMaxWordVectors = True
	limitSentenceLengthsSize = None
	limitSentenceLengths = False
	NLPsequentialInputTypeTrainWordVectors = False
	wordVectorLibraryNumDimensions = 300	#https://spacy.io/models/en#en_core_web_md (300 dimensions)

trainMultipleFiles = False	#can set to true for production (after testing algorithm)
numEpochs = 1
if(numEpochs > 1):
	randomiseFileIndexParse = True
else:
	randomiseFileIndexParse = False

	
#code from ANNtf;
dataset = "wikiXmlDataset"
#if(NLPsequentialInputTypeMinWordVectors):
#	numberOfFeaturesPerWord = 1000	#used by wordToVec
paddingTagIndex = 0.0	#not used
if(debugUseSmallSequentialInputDataset):
	dataset4FileNameXstart = "Xdataset4PartSmall"
else:
	dataset4FileNameXstart = "Xdataset4Part"
xmlDatasetFileNameEnd = ".xml"
def loadDataset(fileIndex, textualDatasetLoadPerformProcessing=True):

	global numberOfFeaturesPerWord
	global paddingTagIndex
	
	datasetNumFeatures = 0
	datasetNumClasses = 0
	
	fileIndexStr = str(fileIndex).zfill(4)
	if(dataset == "POStagSequence"):
		datasetType1FileNameX = dataset1FileNameXstart + fileIndexStr + datasetFileNameXend
		datasetType1FileNameY = dataset1FileNameYstart + fileIndexStr + datasetFileNameYend
	elif(dataset == "POStagSentence"):
		datasetType3FileNameX = dataset3FileNameXstart + fileIndexStr + datasetFileNameXend		
	elif(dataset == "SmallDataset"):
		if(trainMultipleFiles):
			datasetType2FileName = dataset2FileNameStart + fileIndexStr + datasetFileNameEnd
		else:
			datasetType2FileName = dataset2FileName
	elif(dataset == "wikiXmlDataset"):
		datasetType4FileName = dataset4FileNameXstart + fileIndexStr + xmlDatasetFileNameEnd

	numberOfLayers = 0
	if(dataset == "POStagSequence"):
		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = ANNtf2_loadDataset.loadDatasetType1(datasetType1FileNameX, datasetType1FileNameY, addOnlyPriorUnidirectionalPOSinputToTrain)
		if(trainDataIncludesSentenceOutOfBoundsIndex):
			datasetNumClasses = datasetNumClasses + 1
	elif(dataset == "POStagSentence"):
		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = ANNtf2_loadDataset.loadDatasetType3(datasetType3FileNameX, generatePOSunambiguousInput, onlyAddPOSunambiguousInputToTrain, limitSentenceLengthsSize)
	elif(dataset == "SmallDataset"):
		datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = ANNtf2_loadDataset.loadDatasetType2(datasetType2FileName, datasetClassColumnFirst)
		numberOfFeaturesPerWord = None
		paddingTagIndex = None
	elif(dataset == "wikiXmlDataset"):
		articles = ANNtf2_loadDataset.loadDatasetType4(datasetType4FileName, limitSentenceLengths, limitSentenceLengthsSize, NLPsequentialInputTypeTrainWordVectors, NLPsequentialInputTypeTokeniseWords)
		if(textualDatasetLoadPerformProcessing):
			numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = ANNtf2_loadDataset.convertArticlesTreeToSentencesWordVectors(articles, limitSentenceLengthsSize)

	if((dataset == "wikiXmlDataset") and not textualDatasetLoadPerformProcessing):
		return articles
	else:
		return numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y


def trainSequentialInput(trainMultipleFiles=False):
	
	if(algorithmATNLP == "normaliseInputVectors"):
		ATNLPtf_normalisation.constructPOSdictionary()	#required for ATNLPtf_normalisation:ATNLPtf_getAllPossiblePosTags.getAllPossiblePosTags(word)
	#elif(algorithmATNLP == "generateGraph"):	#use spacy POS detection (whole sentence) instead of nltk pos detection
	
	networkIndex = 1	#assert numberOfNetworks = 1
	fileIndexTemp = 0	#assert trainMultipleFiles = False

	#configure optional parameters;
	if(trainMultipleFiles):
		minFileIndex = fileIndexFirst
		maxFileIndex = fileIndexLast
	else:
		minFileIndex = 0
		maxFileIndex = 0
	
	for e in range(numEpochs):

		print("epoch e = ", e)
	
		#fileIndex = 0
		#trainMultipleFiles code;
		if(randomiseFileIndexParse):
			fileIndexShuffledArray = ANNtf2_loadDataset.generateRandomisedIndexArray(fileIndexFirst, fileIndexLast)
		for f in range(minFileIndex, maxFileIndex+1):
			if(randomiseFileIndexParse):
				fileIndex = fileIndexShuffledArray[f]
			else:
				fileIndex = f

			#ATNLP specific code;
							
			articles = loadDataset(fileIndex, textualDatasetLoadPerformProcessing=False)	#do not perform processing of textual dataset during load (word vector extraction)
								
			#print("articles = ", articles)
			#print("listDimensions(articles) = ", listDimensions(articles))
			
			if(supportBatchAndMultiAbstractionLevelProcessing):
				ATNLPtf_processingBatchAndMultiAbstractionLevel.processingBatchAndMultiAbstractionLevel(articles)
			else:
				processingSimple(articles)
					
						
def processingSimple(articles):
	if(NLPsequentialInputTypeMaxWordVectors):
		#flatten any higher level abstractions defined in NLPsequentialInputTypeMax down to word vector lists (sentences);
		articles = ANNtf2_loadDataset.flattenNestedListToSentences(articles)
						
	for sentenceIndex, sentence in enumerate(articles):
		trainSequentialInputNetworkSimple(sentenceIndex, sentence)
	
def trainSequentialInputNetworkSimple(sentenceIndex, textContentList):

	if(algorithmATNLP == "normaliseInputVectors"):
		inputVectorList = ANNtf2_loadDataset.generateWordVectorInputList(textContentList, wordVectorLibraryNumDimensions)	#numberSequentialInputs x inputVecDimensions
		normalisedInputVectorList = ATNLPtf_normalisation.normaliseInputVectorUsingWords(inputVectorList, textContentList)	#normalise length
		#network propagation (TODO);
	elif(algorithmATNLP == "generateGraph"):
		ATNLPtf_graph.generateGraph(sentenceIndex, textContentList)	#!NLPsequentialInputTypeTokeniseWords: textContentList=sentence
		
	
if __name__ == "__main__":
	trainSequentialInput(trainMultipleFiles=trainMultipleFiles)

