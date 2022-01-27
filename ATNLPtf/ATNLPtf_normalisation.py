"""ATNLPtf_normalisation.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATNLPtf_main.py

# Usage:
see ATNLPtf_main.py

# Description:
ATNLP normalisation - create normalised patches from input vectors (1D axis transformation/resize at NLP/POS keypoints)

"""

import spacy
import tensorflow as tf
import numpy as np
import copy
import ATNLPtf_getAllPossiblePosTags

patchNormalisationSize = 100
patchNormalisationSizeAntialias = True

sentenceNormalisationDelimiterPOStags = [".", "IN", ",", ";"]	#IN=prep	#CHECKTHIS

spacyWordVectorGenerator = spacy.load('en_core_web_md')	#spacy.load('en_core_web_lg')

#should be defined as preprocessor defs (non-variable);
ATNLPsequentialInputTypeCharacters = 0
ATNLPsequentialInputTypeWords = 1
ATNLPsequentialInputTypeSentences = 2
ATNLPsequentialInputTypeParagraphs = 3	
ATNLPsequentialInputTypeArticles = 4
ATNLPsequentialInputTypes = ["characters", "words", "sentences", "paragraphs"]
ATNLPsequentialInputNumberOfTypes = len(ATNLPsequentialInputTypes)

def constructPOSdictionary():
	ATNLPtf_getAllPossiblePosTags.constructPOSdictionary()	#required for getKeypoints
 
#code from AEANNtf;
def flattenNestedListToSentences(articles):
	articlesFlattened = []
	nestedList = articles
	for ATNLPsequentialInputTypeIndex in range(ATNLPsequentialInputTypeArticles, ATNLPsequentialInputTypeSentences, -1):
		#print("ATNLPsequentialInputTypeIndex = ", ATNLPsequentialInputTypeIndex)
		flattenedList = []
		for content in nestedList:
			flattenedList.extend(content)
		#print("flattenedList = ", flattenedList)
		nestedList = flattenedList	#for recursion
	ATNLPsequentialInputTypeMaxTemp = ATNLPsequentialInputTypeWords
	articlesFlattened = nestedList
	#print("articles = ", articles)
	#print("listDimensions(articlesFlattened) = ", listDimensions(articlesFlattened))
	return articlesFlattened
			
def generateWordVectorInputList(textContentList, ATNLPsequentialInputDimensions):
	inputVectorList = []
	for word in textContentList:
		#print("word = ", word)
		doc = spacyWordVectorGenerator(word)
		wordVectorList = doc[0].vector	#verify type numpy
		wordVector = np.array(wordVectorList)
		#print("word = ", word, " wordVector = ", wordVector)
		inputVectorList.append(wordVector)
	return inputVectorList
	
#code from AEANNtf;
def generateRandomisedIndexArray(indexFirst, indexLast, arraySize=None):
	fileIndexArray = np.arange(indexFirst, indexLast+1, 1)
	#print("fileIndexArray = " + str(fileIndexArray))
	if(arraySize is None):
		np.random.shuffle(fileIndexArray)
		fileIndexRandomArray = fileIndexArray
	else:
		fileIndexRandomArray = random.sample(fileIndexArray.tolist(), arraySize)
	
	#print("fileIndexRandomArray = " + str(fileIndexRandomArray))
	return fileIndexRandomArray
			
def normaliseInputVectorUsingWords(inputVectorList, textWordList):
	#generates sets of normalised 
	print("textWordList = ", textWordList)
	keypointsList = getKeypoints(textWordList)
	normalisedInputVectorList = normaliseInputVectorUsingKeypoints(inputVectorList, keypointsList)
	return normalisedInputVectorList

def getKeypoints(textWordList):	
	#ATOR terminology; keypoint = feature
	keypointsList = []
	
	textPosList = []
	for word in textWordList:
		#print("word = ", word)
		posValues = ATNLPtf_getAllPossiblePosTags.getAllPossiblePosTags(word)
		textPosList.append(posValues)
		#print(word, ", ", posValues)
	
	for wordIndex, posValues in enumerate(textPosList):	
		foundKeypoint = False
		for posValue in posValues:
			if(posValue in sentenceNormalisationDelimiterPOStags):
				foundKeypoint = True
		
		if(wordIndex == 0):
			foundKeypoint = True	#always set first word in sentence as a keypoint
			
		if(foundKeypoint):
			#print("foundKeypoint: wordIndex = ", wordIndex)
			keypointsList.append(True)
		else:
			keypointsList.append(False)
			
	return keypointsList

def normaliseInputVectorUsingKeypoints(inputVectorList, keypointsList):
	normalisedInputVectorList = []

	#create a 1D tf image (of inputvector-dimension channels);
	inputVectorImage = tf.convert_to_tensor(inputVectorList, dtype=tf.float32)
	
	#print("sentence length = ", len(inputVectorList))
	
	#for every permutation in keypointsList, create a normalised representation of inputVectorList (CNN input)
		#stretch width to a certain amount for use as CNN input
	for kp1Index, kp1 in enumerate(keypointsList):
		for kp2Index, kp2 in enumerate(keypointsList):
			if(kp2Index > kp1Index):
				if(kp1 and kp2):
					#print("kp1Index = ", kp1Index)
					#print("kp2Index = ", kp2Index)

					inputVectorImageSub = inputVectorImage[kp1Index:kp2Index]	#extract patch

					inputVectorImageSub = tf.expand_dims(inputVectorImageSub, 0)	#add an empty batch dimension
					inputVectorImageSub = tf.expand_dims(inputVectorImageSub, 2)	#add an empty y dimension
					#print("inputVectorImageSub.shape = ", inputVectorImageSub.shape)

					normalisedInputVector = tf.image.resize(inputVectorImageSub, (patchNormalisationSize, 1), antialias=patchNormalisationSizeAntialias)	#normalise the input vector
					#print("normalisedInputVector.shape = ", normalisedInputVector.shape)

					normalisedInputVectorList.append(normalisedInputVector)
	
	print("normalisedInputVectorList length = ", len(normalisedInputVectorList))
	
	return normalisedInputVectorList
