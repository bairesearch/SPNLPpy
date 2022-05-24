"""ATNLPtf_processingBatchAndMultiAbstractionLevel.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ATNLPtf_main.py

# Usage:
see ATNLPtf_main.py

# Description:
ATNLP Processing Batch And Multi Abstraction Level

Incomplete: draft only - batches are not currently processed/normalised in parallel (retained for source base compatibility)

"""

import ATNLPtf_normalisation

#code from AEANNtf;

#performance enhancements for development environment only: 
if(trainMultipleFiles):
	batchSize = 10	#max 1202	#defined by wiki database extraction size
else:
	batchSize = 1	#def:False	#switch increases performance during development	#eg data-simple-POStagSentence-smallBackup

#should be defined as preprocessor defs (non-variable);
NLPsequentialInputTypeCharacters = 0
NLPsequentialInputTypeWords = 1
NLPsequentialInputTypeSentences = 2
NLPsequentialInputTypeParagraphs = 3	
NLPsequentialInputTypeArticles = 4
NLPsequentialInputTypes = ["characters", "words", "sentences", "paragraphs"]
ATNLPsequentialInputNumberOfTypes = len(NLPsequentialInputTypes)

NLPsequentialInputType = NLPsequentialInputTypeWords
NLPsequentialInputTypeName = NLPsequentialInputTypes[NLPsequentialInputType] #eg "words"
NLPsequentialInputTypeMax = NLPsequentialInputTypeParagraphs	#0:characters, 1:words, 2:sentences, 3:paragraphs
NLPsequentialInputTypeMinWordVectors = True	#only train network using word vector level input (no lower abstractions)	#lookup input vectors for current level input type (e.g. if words, using a large word2vec database), else generate input vectors using lowever level AEANN network #not supported by NLPsequentialInputType=characters
NLPsequentialInputTypeMaxWordVectors = True	#enable during testing	#only train network using word vector level input (no higher abstractions)	#will flatten any higher level abstractions defined in NLPsequentialInputTypeMax down to word vector lists (sentences)
NLPsequentialInputTypeTrainWordVectors = (not NLPsequentialInputTypeMinWordVectors)

limitSentenceLengths = True
if(limitSentenceLengths): 
	limitSentenceLengthsSize = [10, 10, 10, 10]	#temporarily reduce input size for debug/processing speed
else:
	limitSentenceLengthsSize = [100, 100, 100, 100]	#implementation does not require this to be equal for each type (n_h[0] does not have to be identical for each network)	#inputs shorter than max length are padded

if(AEANNtf_algorithm.networkEqualInputVectorDimensions):
	if(NLPsequentialInputTypeMinWordVectors):
		wordVectorLibraryNumDimensions = ANNtf2_loadDataset.wordVectorLibraryNumDimensions	#https://spacy.io/models/en#en_core_web_md (300 dimensions)
		inputVectorNumDimensions = wordVectorLibraryNumDimensions
	else:
		asciiNumberCharacters = 128
		inputVectorNumDimensions = asciiNumberCharacters
	NLPsequentialInputTypesVectorDimensions = [inputVectorNumDimensions, inputVectorNumDimensions, inputVectorNumDimensions, inputVectorNumDimensions]
else:
	print("ATNLPtf_main error; only AEANNtf_algorithmSequentialInput.networkEqualInputVectorDimensions is currently coded")
	exit()
	#NLPsequentialInputTypesVectorDimensions = [asciiNumberCharacters, wordVectorNumDimensions, getNumNetworkNodes(limitSentenceLengthsSize[NLPsequentialInputTypeWords], getNumNetworkNodes(limitSentenceLengthsSize[NLPsequentialInputTypeSentences])]

#maxWordLength=limitSentenceLengthsSize[0]	#in characters
#maxSentenceLength=limitSentenceLengthsSize[1]	#in words
#maxParagraphLength=limitSentenceLengthsSize[2]	#in sentences
#maxCorpusLength=limitSentenceLengthsSize[3]	#in paragraphs

if(NLPsequentialInputTypeMinWordVectors):
	NLPsequentialInputTypeMin = NLPsequentialInputTypeWords
	trainMultipleNetworks = False
	#numberOfNetworks = NLPsequentialInputType+1-NLPsequentialInputTypeWords	#use word vectors from preexisting database
else:
	trainMultipleNetworks = True
	NLPsequentialInputTypeMin = 0
	#numberOfNetworks = NLPsequentialInputType+1
numberOfNetworks = AEANNsequentialInputNumberOfTypes	#full, even though not all networks may be used


def processingBatchAndMultiAbstractionLevel(articles)

	if(NLPsequentialInputTypeMaxWordVectors):
		#flatten any higher level abstractions defined in AEANNsequentialInputTypeMax down to word vector lists (sentences);
		articles = ATNLPtf_normalisation.flattenNestedListToSentences(articles)
							
	#generate random batches/samples from/of articles set;
	batchesList = []	#all batches
	for batchIndex in range(int(trainingSteps)):
		#FUTURE: is randomisation an appropriate batch generation algorithm (rather than guaranteeing every sample is added to a batch)?
		sampleIndexFirst = 0
		sampleIndexLast = len(articles)-1
		sampleIndexShuffledArray = ANNtf2_loadDataset.generateRandomisedIndexArray(sampleIndexFirst, sampleIndexLast, arraySize=batchSize)
		paragraphs = [articles[i] for i in sampleIndexShuffledArray]
		batchesList.append(paragraphs)

	print("listDimensions(batchesList) = ", listDimensions(batchesList))

	for batchIndex, batch in enumerate(batchesList):
		#print("listDimensions(batch) = ", listDimensions(batch))
		#print("batch = ", batch)

		NLPsequentialInputTypeMaxTemp = None
		batchNestedList = batch
		#print("listDimensions(batchNestedList) = ", listDimensions(batchNestedList))
		NLPsequentialInputTypeMaxTemp = NLPsequentialInputTypeMax

		if(NLPsequentialInputTypeMaxWordVectors and NLPsequentialInputTypeMinWordVectors):
			trainSequentialInputNetwork(batchIndex, NLPsequentialInputTypeWords, batchNestedList, None, optimizer)
		else:	
			layerInputVectorListGenerated = trainSequentialInputNetworkRecurse(batchIndex, NLPsequentialInputTypeMaxTemp, batchNestedList, optimizer)	#train all AEANN networks (at each layer of abstraction)
			#trainSequentialInputNetwork(batchIndex, NLPsequentialInputTypeWords, batchNestedList, layerInputVectorListGenerated, optimizer)	#CHECKTHIS is not required

def trainSequentialInputNetworkRecurse(batchIndex, NLPsequentialInputTypeIndex, batchNestedList, optimizer):

	higherLayerInputVectorList = []
	maxNumberNestedListElements = limitSentenceLengthsSize[NLPsequentialInputTypeIndex]
	for nestedListElementIndex in range(maxNumberNestedListElements):
		print("nestedListElementIndex = ", nestedListElementIndex)
		batchNestedListElement = []	#batched
		for nestedList in batchNestedList:
			if(nestedListElementIndex < len(nestedList)):
				batchNestedListElement.append(nestedList[nestedListElementIndex])
			else:	
				emptyList = []
				batchNestedListElement.append(emptyList)
		#print("listDimensions(batchNestedListElement) = ", listDimensions(batchNestedListElement))
		layerInputVectorListGenerated = None
		if(NLPsequentialInputTypeIndex > NLPsequentialInputTypeMin):
			layerInputVectorListGenerated = trainSequentialInputNetworkRecurse(batchIndex, NLPsequentialInputTypeIndex-1, batchNestedListElement, optimizer)	#train lower levels before current level
		higherLayerInputVector = trainSequentialInputNetwork(batchIndex, NLPsequentialInputTypeIndex, batchNestedListElement, layerInputVectorListGenerated, optimizer)
		higherLayerInputVectorList.append(higherLayerInputVector)
		
	return higherLayerInputVectorList	#shape: numberSequentialInputs x batchSize x inputVecDimensions
	
def trainSequentialInputNetwork(batchIndex, NLPsequentialInputTypeIndex, batchNestedListElement, layerInputVectorListGenerated, optimizer):

	networkIndex = NLPsequentialInputTypeIndex

	batchInputVectorList = []
	if(layerInputVectorListGenerated is None):
		#print("!layerInputVectorListGenerated")
		for nestedListElementIndex, nestedListElement in enumerate(batchNestedListElement):	#for every batchIndex
			#print("nestedListElementIndex = ", nestedListElementIndex)
			if(nestedListElement):	#verify list is not empty (check is required to compensate for empty parts of batches);
				#print("nestedListElement = ", nestedListElement)
				if(NLPsequentialInputTypeIndex == NLPsequentialInputTypeCharacters):
					#print("NLPsequentialInputTypeIndex == NLPsequentialInputTypeCharacters")
					textCharacterList = nestedListElement	#batched
					#inputVectorList = AEANNtf_algorithm.generateCharacterVectorInputList(textCharacterList, NLPsequentialInputTypesVectorDimensions[NLPsequentialInputTypeIndex], limitSentenceLengthsSize[NLPsequentialInputTypeIndex])	#numberSequentialInputs x inputVecDimensions
	 				print("NLPsequentialInputTypeCharacters not supported by ATNLP")
					exit()
				elif(NLPsequentialInputTypeIndex == NLPsequentialInputTypeWords):
					#print("NLPsequentialInputTypeIndex == NLPsequentialInputTypeWords")
					if(NLPsequentialInputTypeMinWordVectors):		#implied true (because layerInputVectorListGenerated is None)	
						#print("NLPsequentialInputTypeMinWordVectors")
						textWordList = nestedListElement	#batched
						inputVectorList = ATNLPtf_normalisation.generateWordVectorInputList(textWordList, NLPsequentialInputTypesVectorDimensions[NLPsequentialInputTypeIndex])	#numberSequentialInputs x inputVecDimensions
			else:
				inputVectorList = AEANNtf_algorithm.generateBlankVectorInputList(NLPsequentialInputTypesVectorDimensions[NLPsequentialInputTypeIndex], limitSentenceLengthsSize[NLPsequentialInputTypeIndex])	#empty sequence (seq length = 0)
			#print("inputVectorList = ", inputVectorList)
			batchInputVectorList.append(inputVectorList)
			
		#print("batchInputVectorList = ", batchInputVectorList)
				
		#samplesInputVector = np.array(list(zip_longest(*samplesInputVectorList, fillvalue=paddingTagIndex))).T	#not required as already padded by AEANNtf_algorithmSequentialInput.generateWordVectorInput/generateCharacterVectorInput
		batchInputVector = np.asarray(batchInputVectorList)	#np.array(samplesInputVectorList)
		print("batchInputVector.shape = ", batchInputVector.shape)
		batchInputVector = tf.convert_to_tensor(batchInputVector, dtype=tf.float32)
		#print("batchInputVector = ", batchInputVector)
		print("!layerInputVectorListGenerated: batchInputVector.shape = ", batchInputVector.shape)	#shape: batchSize x numberSequentialInputs x inputVecDimensions
	else:
		#print("layerInputVectorListGenerated = ", layerInputVectorListGenerated)
		batchInputVector = tf.stack(layerInputVectorListGenerated)	#shape: numberSequentialInputs x batchSize x inputVecDimensions
		print("batchInputVector.shape = ", batchInputVector.shape)
		batchInputVector = tf.transpose(batchInputVector, (1, 0, 2))	#shape: batchSize x numberSequentialInputs x inputVecDimensions
		#use existing inputVectors generated from lower layer
		#print("layerInputVectorListGenerated: batchInputVector = ", batchInputVector)
		print("layerInputVectorListGenerated: batchInputVector.shape = ", batchInputVector.shape)
	
	#normalise input vectors
	for inputVectorList in batchInputVectorList:	#for every batchIndex
		normalisedInputVectorList = ATNLPtf_normalisation.normaliseInputVectorUsingWords(inputVectorList, textWordList)	#normalise length
		

	#network propagation;
	
	higherLayerInputVector = AEANNtf_algorithm.neuralNetworkPropagationTestGenerateHigherLayerInputVector(networkIndex, batchInputVector)	#batchSize x inputVecDimensions		#propagate through network and return activation levels of all neurons	
	return higherLayerInputVector
