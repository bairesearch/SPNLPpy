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

sentenceNormalisationDelimiterPOStags = [".", "CC", "IN", "TO", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", ",", ";"]	#CHECKTHIS	#require to detect rcmod (that/which etc), additional punctuation ('', (, ), ,, --, ., :), RP particle, etc?	#keypoint/feature detection
	#nltk pos tags
	
def constructPOSdictionary():
	ATNLPtf_getAllPossiblePosTags.constructPOSdictionary()	#required for getKeypoints
 		
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
	
	print("keypointsList = ", keypointsList)
		
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
