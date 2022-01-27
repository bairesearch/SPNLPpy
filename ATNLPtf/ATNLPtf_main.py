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
import ATNLPtf_normalisation


supportBatchAndMultiAbstractionLevelProcessing = False		#batches are not currently processed/normalised in parallel (retained for source base compatibility);
if(supportBatchAndMultiAbstractionLevelProcessing):
	import ATNLPtf_processingBatchAndMultiAbstractionLevel
	ATNLPsequentialInputTypeMinWordVectors = ATNLPtf_processingBatchAndMultiAbstractionLevel.ATNLPsequentialInputTypeMinWordVectors
	ATNLPsequentialInputTypeMaxWordVectors = ATNLPtf_processingBatchAndMultiAbstractionLevel.ATNLPsequentialInputTypeMaxWordVectors
	ATNLPsequentialInputTypesMaxLength = ATNLPtf_processingBatchAndMultiAbstractionLevel.ATNLPsequentialInputTypesMaxLength
	useSmallSentenceLengths = ATNLPtf_processingBatchAndMultiAbstractionLevel.useSmallSentenceLengths
	ATNLPsequentialInputTypeTrainWordVectors = ATNLPtf_processingBatchAndMultiAbstractionLevel.ATNLPsequentialInputTypeTrainWordVectors	
if(not supportBatchAndMultiAbstractionLevelProcessing):
	#mandatory for !supportBatchAndMultiAbstractionLevelProcessing;
	ATNLPsequentialInputTypeMinWordVectors = True
	ATNLPsequentialInputTypeMaxWordVectors = True
	ATNLPsequentialInputTypesMaxLength = None
	useSmallSentenceLengths = False
	ATNLPsequentialInputTypeTrainWordVectors = False
	wordVectorLibraryNumDimensions = 300	#https://spacy.io/models/en#en_core_web_md (300 dimensions)

trainMultipleFiles = False	#can set to true for production (after testing algorithm)
numEpochs = 1
if(numEpochs > 1):
	randomiseFileIndexParse = True
else:
	randomiseFileIndexParse = False

	
#code from ANNtf/AEANNtf;
dataset = "wikiXmlDataset"
#if(ATNLPsequentialInputTypeMinWordVectors):
#	numberOfFeaturesPerWord = 1000	#used by wordToVec
paddingTagIndex = 0.0	#not used
debugUseSmallSequentialInputDataset = False
if(debugUseSmallSequentialInputDataset):
	dataset4FileNameStart = "Xdataset4PartSmall"
else:
	dataset4FileNameStart = "Xdataset4Part"
xmlDatasetFileNameEnd = ".xml"
def loadDataset(fileIndex):

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
		datasetType4FileName = dataset4FileNameStart + fileIndexStr + xmlDatasetFileNameEnd
			
	numberOfLayers = 0
	if(dataset == "POStagSequence"):
		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_xTemp, train_yTemp, test_xTemp, test_yTemp = ANNtf2_loadDataset.loadDatasetType1(datasetType1FileNameX, datasetType1FileNameY)
	elif(dataset == "POStagSentence"):
		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_xTemp, train_yTemp, test_xTemp, test_yTemp = ANNtf2_loadDataset.loadDatasetType3(datasetType3FileNameX, generatePOSunambiguousInput, onlyAddPOSunambiguousInputToTrain, useSmallSentenceLengths)
	elif(dataset == "SmallDataset"):
		datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_xTemp, train_yTemp, test_xTemp, test_yTemp = ANNtf2_loadDataset.loadDatasetType2(datasetType2FileName, datasetClassColumnFirst)
		numberOfFeaturesPerWord = None
		paddingTagIndex = None
	elif(dataset == "wikiXmlDataset"):
		articles = ANNtf2_loadDataset.loadDatasetType4(datasetType4FileName, ATNLPsequentialInputTypesMaxLength, useSmallSentenceLengths,  ATNLPsequentialInputTypeTrainWordVectors)

	if(dataset == "wikiXmlDataset"):
		return articles
	else:
		return numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_xTemp, train_yTemp, test_xTemp, test_yTemp


def trainSequentialInput(trainMultipleFiles=False):
	
	ATNLPtf_normalisation.constructPOSdictionary()	#required for ATNLPtf_normalisation:ATNLPtf_getAllPossiblePosTags.getAllPossiblePosTags(word)
	
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
			fileIndexShuffledArray = generateRandomisedIndexArray(fileIndexFirst, fileIndexLast)
		for f in range(minFileIndex, maxFileIndex+1):
			if(randomiseFileIndexParse):
				fileIndex = fileIndexShuffledArray[f]
			else:
				fileIndex = f

			#ATNLP specific code;
							
			articles = loadDataset(fileIndex)
					
			#numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y
			
			#print("articles = ", articles)
			#print("listDimensions(articles) = ", listDimensions(articles))
			
			if(supportBatchAndMultiAbstractionLevelProcessing):
				ATNLPtf_processingBatchAndMultiAbstractionLevel.processingBatchAndMultiAbstractionLevel(articles)
			else:
				processingSimple(articles)
				
def processingSimple(articles):
	if(ATNLPsequentialInputTypeMaxWordVectors):
		#flatten any higher level abstractions defined in ATNLPsequentialInputTypeMax down to word vector lists (sentences);
		articles = ATNLPtf_normalisation.flattenNestedListToSentences(articles)
				
	for sentence in articles:
		trainSequentialInputNetworkSimple(sentence)
	
def trainSequentialInputNetworkSimple(textContentList):

	inputVectorList = ATNLPtf_normalisation.generateWordVectorInputList(textContentList, wordVectorLibraryNumDimensions)	#numberSequentialInputs x inputVecDimensions

	#normalise input vectors
	normalisedInputVectorList = ATNLPtf_normalisation.normaliseInputVectorUsingWords(inputVectorList, textContentList)	#normalise length
		
	#network propagation (TODO);

	
if __name__ == "__main__":
	trainSequentialInput(trainMultipleFiles=trainMultipleFiles)

