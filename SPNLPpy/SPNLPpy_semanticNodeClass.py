"""SPNLPpy_semanticNodeClass.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SPNLPpy_main.py

# Usage:
see SPNLPpy_main.py

# Description:
SPNLP Semantic Node Class

See GIA/GIAentityNodeClass.hpp for template

"""

import numpy as np

actionDetectionAnyCandidateVerbPOS = False	#if the word can
if(actionDetectionAnyCandidateVerbPOS):
	import SPNLPpy_getAllPossiblePosTags
	nltkPosTagTypeVerb = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
	#FUTURE: getAllPossiblePosTags for a word (context ignored) with spacy library

GIA_ADD_ARTIFICIAL_AUXILIARY_FOR_ALL_PROPERTIES_AND_DEFINITIONS = False	#this feature is not currently supported by SPNLPpy_semanticGraph (GIA C++ implementation only)


#sync with LRPglobalsDefs.hpp;

GRAMMATICAL_DETERMINER_DEFINITE = "the"
GRAMMATICAL_DETERMINER_INDEFINITE_SINGULAR = "a"
GRAMMATICAL_DETERMINER_INDEFINITE_PLURAL = "some"
GRAMMATICAL_DETERMINER_INDEFINITE_SINGULAR_FIRST_LETTER_VOWEL = "an"
GRAMMATICAL_DETERMINER_DEFINITE_EACH = "each"
GRAMMATICAL_DETERMINER_DEFINITE_EVERY = "every"
GRAMMATICAL_DETERMINER_INDEFINITE_ALL = "all"
GRAMMATICAL_DETERMINER_INDEFINITE_NUMBER_OF_TYPES = (2)
grammaticalDeterminerIndefiniteArray = [GRAMMATICAL_DETERMINER_INDEFINITE_SINGULAR, GRAMMATICAL_DETERMINER_INDEFINITE_SINGULAR_FIRST_LETTER_VOWEL]	#NB this intentionally discludes GRAMMATICAL_DETERMINER_INDEFINITE_PLURAL "some" as this is handled the same as a definite determinier by GIA2 POS tag system
GRAMMATICAL_DETERMINER_DEFINITE_NUMBER_OF_TYPES = (3)
grammaticalDeterminerDefiniteArray = [GRAMMATICAL_DETERMINER_DEFINITE, GRAMMATICAL_DETERMINER_DEFINITE_EACH, GRAMMATICAL_DETERMINER_DEFINITE_EVERY]	#NB this intentionally discludes GRAMMATICAL_DETERMINER_INDEFINITE_PLURAL "some" as this is handled the same as a definite determinier by GIA2 POS tag system

GRAMMATICAL_AUXILIARY_BEING_PRESENT_SINGULAR = "is"

LRP_SHARED_ENTITY_TYPE_UNDEFINED = (-1)
LRP_SHARED_ENTITY_TYPE_NETWORK_INDEX = (0)
LRP_SHARED_ENTITY_TYPE_SUBSTANCE = (1)
LRP_SHARED_ENTITY_TYPE_CONCEPT = (2)
LRP_SHARED_ENTITY_TYPE_ACTION = (3)
LRP_SHARED_ENTITY_TYPE_CONDITION = (4)
LRP_SHARED_ENTITY_TYPE_PROPERTY = (5)
LRP_SHARED_ENTITY_TYPE_DEFINITION = (6)
LRP_SHARED_ENTITY_TYPE_QUALITY = (7)
LRP_SHARED_ENTITY_NUMBER_OF_TYPES = (8)


#sync with GIAentityNodeClass.hpp;

GIA_ENTITY_TYPE_UNDEFINED = (LRP_SHARED_ENTITY_TYPE_UNDEFINED)
GIA_ENTITY_TYPE_NETWORK_INDEX = (LRP_SHARED_ENTITY_TYPE_NETWORK_INDEX)
GIA_ENTITY_TYPE_SUBSTANCE = (LRP_SHARED_ENTITY_TYPE_SUBSTANCE)
GIA_ENTITY_TYPE_CONCEPT = (LRP_SHARED_ENTITY_TYPE_CONCEPT)
GIA_ENTITY_TYPE_ACTION = (LRP_SHARED_ENTITY_TYPE_ACTION)
GIA_ENTITY_TYPE_CONDITION = (LRP_SHARED_ENTITY_TYPE_CONDITION)
GIA_ENTITY_TYPE_PROPERTY = (LRP_SHARED_ENTITY_TYPE_PROPERTY)
GIA_ENTITY_TYPE_DEFINITION = (LRP_SHARED_ENTITY_TYPE_DEFINITION)
GIA_ENTITY_TYPE_QUALITY = (LRP_SHARED_ENTITY_TYPE_QUALITY)
GIA_ENTITY_NUMBER_OF_TYPES = (LRP_SHARED_ENTITY_NUMBER_OF_TYPES)

entityTypesIsRelationshipArray = [False, False, False, True, True, True, True, False]
entityTypesAutomaticallyUpgradeUponInstanceSelectionArray = [False, True, True, True, True, True, True, False]
entityTypesIsActionOrConditionRelationshipArray = [False, False, False, True, True, False, False, False]
entityTypesIsPropertyOrDefinitionRelationshipArray = [False, False, False, False, False, True, True, False]


GIA_ENTITY_NUMBER_OF_VECTOR_CONNECTION_TYPES = (12)
GIA_ENTITY_VECTOR_CONNECTION_TYPE_ACTION = (0)
GIA_ENTITY_VECTOR_CONNECTION_TYPE_ACTION_REVERSE = (1)
GIA_ENTITY_VECTOR_CONNECTION_TYPE_CONDITION = (2)
GIA_ENTITY_VECTOR_CONNECTION_TYPE_CONDITION_REVERSE = (3)
GIA_ENTITY_VECTOR_CONNECTION_TYPE_PROPERTY = (4)
GIA_ENTITY_VECTOR_CONNECTION_TYPE_PROPERTY_REVERSE = (5)
GIA_ENTITY_VECTOR_CONNECTION_TYPE_DEFINITION = (6)
GIA_ENTITY_VECTOR_CONNECTION_TYPE_DEFINITION_REVERSE = (7)
GIA_ENTITY_VECTOR_CONNECTION_TYPE_RELATIONSHIP_SUBJECT = (8)
GIA_ENTITY_VECTOR_CONNECTION_TYPE_RELATIONSHIP_OBJECT = (9)
GIA_ENTITY_VECTOR_CONNECTION_TYPE_INSTANCE = (10)
GIA_ENTITY_VECTOR_CONNECTION_TYPE_INSTANCE_REVERSE = (11)	

RELATION_ENTITY_HAVE = "have"
RELATION_ENTITY_BE = "be"	#eg x is y
multiwordRelationshipSpecialList = [[RELATION_ENTITY_BE, GRAMMATICAL_DETERMINER_INDEFINITE_SINGULAR]]

#https://github.com/explosion/spaCy/blob/master/spacy/glossary.py
#Universal POS Tags (https://universaldependencies.org/u/pos/)
spacyPosTagTypeNoun = ["NOUN"]	#Substance
spacyPosTagTypeVerb = ["VERB"]	#Action
spacyPosTagTypePreposition = ["ADP"]	#Condition
spacyPosTagTypeAdverb= ["ADV"]	#ActionQuality
spacyPosTagTypeAdjective = ["ADJ"]	#SubstanceQuality
#not currently used;
spacyPosTagTypeCoordinatingConjunction = ["CCONJ"]
spacyPosTagTypeDeterminer = ["DET"]
spacyPosTagTypeAuxiliary = ["AUX"]
spacyPosTagTypeInterjection = ["INTJ"]
spacyPosTagTypeNumber = ["NUM"]
spacyPosTagTypeParticle = ["PART"]
spacyPosTagTypePronoun = ["PRON"]
spacyPosTagTypePropernoun = ["PROPN"]
spacyPosTagTypePunctuation = ["PUNCT"]
spacyPosTagTypeSubordinatingConjunction = ["SCONJ"]
spacyPosTagTypeSymbol = ["SYM"]
spacyPosTagTypeOther = ["X"]
	#POS tags (English)
	#spacyPosTagTypeNoun = ["NN", "NNP", "NNPS", "NNS"]	#Substance
	#spacyPosTagTypeVerb = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]	#Action
	#spacyPosTagTypePreposition = ["IN"]	#Condition
	#spacyPosTagTypeAdverb= ["RB", "RBR", "RBS", "RP"]	#ActionQuality
	#spacyPosTagTypeAdjective = ["JJ", "JJR", "JJS"]	#SubstanceQuality
	#spacyPosTagTypeSymbol = [".", ",", "-LRB-", "-RRB-", "``", "\"\"", "''", ",", "$", "#", "AFX"]	#grammar
	#spacyPosTagTypeConjunction = ["CC"]
	#spacyPosTagTypeNumber = ["CD"]
	#spacyPosTagTypeDeterminer = ["DT"]
	#spacyPosTagTypeQuestion = ["WDT", "WP", "WP$", "WRB"]

firstIndexInConnectionList = 0	#intrasentence semantic node connection type lists typically contain only 1 connection (during dev)	#CHECKTHIS

class SemanticNode:
	def __init__(self, instanceID, entityName, wordOrig, wordVector, entityType):
		#GIA Internal Entity Referencing;
		self.instanceID = instanceID
		
		#GIA Entity Name;
		self.entityName = entityName	#lemma
		self.wordOrig = wordOrig	#word
		self.wordVector = wordVector	#SPNLP introduction to better support /aliases (not yet implemented in GIA)

		#GIA Entity Type;
		self.entityType = entityType
		#self.referenceSetDelimiter = False	#if entityIsRelationship only
		#self.subreferenceSetDelimiter = False	#if entityIsRelationship only
		
		#GIA Connections;
		self.entityVectorConnectionsArray = [[] for i in range(GIA_ENTITY_NUMBER_OF_VECTOR_CONNECTION_TYPES)]	#NO: [[]]*GIA_ENTITY_NUMBER_OF_VECTOR_CONNECTION_TYPES	#allows for generic coding
		self.actionNodeList = self.entityVectorConnectionsArray[GIA_ENTITY_VECTOR_CONNECTION_TYPE_ACTION]
		self.actionReverseNodeList = self.entityVectorConnectionsArray[GIA_ENTITY_VECTOR_CONNECTION_TYPE_ACTION_REVERSE]
		self.conditionNodeList = self.entityVectorConnectionsArray[GIA_ENTITY_VECTOR_CONNECTION_TYPE_CONDITION]
		self.conditionReverseNodeList = self.entityVectorConnectionsArray[GIA_ENTITY_VECTOR_CONNECTION_TYPE_CONDITION_REVERSE]
		self.propertyNodeList = self.entityVectorConnectionsArray[GIA_ENTITY_VECTOR_CONNECTION_TYPE_PROPERTY]
		self.propertyReverseNodeList = self.entityVectorConnectionsArray[GIA_ENTITY_VECTOR_CONNECTION_TYPE_PROPERTY_REVERSE]
		self.definitionNodeList = self.entityVectorConnectionsArray[GIA_ENTITY_VECTOR_CONNECTION_TYPE_DEFINITION]
		self.definitionReverseNodeList = self.entityVectorConnectionsArray[GIA_ENTITY_VECTOR_CONNECTION_TYPE_DEFINITION_REVERSE]
		self.relationshipSubjectEntity = self.entityVectorConnectionsArray[GIA_ENTITY_VECTOR_CONNECTION_TYPE_RELATIONSHIP_SUBJECT]
		self.relationshipObjectEntity = self.entityVectorConnectionsArray[GIA_ENTITY_VECTOR_CONNECTION_TYPE_RELATIONSHIP_OBJECT]
		self.instanceNodeList = self.entityVectorConnectionsArray[GIA_ENTITY_VECTOR_CONNECTION_TYPE_INSTANCE]
		self.instanceReverseNodeList = self.entityVectorConnectionsArray[GIA_ENTITY_VECTOR_CONNECTION_TYPE_INSTANCE_REVERSE]


#bool GIAentityNodeClassClass::entityIsRelationship(const GIAentityNode* entity)
def entityIsRelationship(entity):
	return entityTypeIsRelationship(entity.entityType)
def entityTypeIsRelationship(entityType):
	#print("entityType = ", entityType)
	relationshipEntity = False;
	if(entityTypesIsRelationshipArray[entityType]):
		relationshipEntity = True;
	#if(not GIA_ADD_ARTIFICIAL_AUXILIARY_FOR_ALL_PROPERTIES_AND_DEFINITIONS):
	#	if(entityTypesIsPropertyOrDefinitionRelationshipArray[entityType]):
	#		relationshipEntity = True;
	return relationshipEntity;

def entityIsRelationshipAction(entityType):
	result = False
	if(entityType == GIA_ENTITY_TYPE_ACTION):
		result = True
	return result
	
def entityIsRelationshipCondition(entityType):
	result = False
	if(entityType == GIA_ENTITY_TYPE_CONDITION):
		result = True
	return result
	
def getRelationshipEntityConnection(relationshipEntity, sourceEntity, CPsourceNodePosition):
	relationshipEntityConnection = None	
	if(entity.entityType == GIA_ENTITY_TYPE_ACTION):
		if(CPsourceNodePosition == sourceNodePositionFirst):
			relationshipEntityConnectionType = GIA_ENTITY_VECTOR_CONNECTION_TYPE_ACTION_REVERSE
			#relationshipEntityConnection = sourceEntity.actionReverseNodeList[firstIndexInConnectionList]
		elif(CPsourceNodePosition == sourceNodePositionSecond):
			relationshipEntityConnectionType = GIA_ENTITY_VECTOR_CONNECTION_TYPE_ACTION
			#relationshipEntityConnection = sourceEntity.actionNodeList[firstIndexInConnectionList]	
	elif(entity.entityType == GIA_ENTITY_TYPE_CONDITION):
		if(CPsourceNodePosition == sourceNodePositionFirst):
			relationshipEntityConnectionType = GIA_ENTITY_VECTOR_CONNECTION_TYPE_CONDITION_REVERSE
			#relationshipEntityConnection = sourceEntity.conditionReverseNodeList[firstIndexInConnectionList]
		elif(CPsourceNodePosition == sourceNodePositionSecond):
			relationshipEntityConnectionType = GIA_ENTITY_VECTOR_CONNECTION_TYPE_CONDITION
			#relationshipEntityConnection = sourceEntity.conditionNodeList[firstIndexInConnectionList]
	
	if(relationshipEntityConnection == None):
		print("getRelationshipEntityConnection error: getRelationshipEntityConnection not found")
		exit()

	relationshipEntityConnection = sourceEntity.entityVectorConnectionsArray[relationshipEntityConnectionType]

	return relationshipEntityConnection, relationshipEntityConnectionType
	
def identifyEntityType(syntacticalNode):
	
	if(syntacticalNode.entityType == GIA_ENTITY_TYPE_UNDEFINED):
		entityType = GIA_ENTITY_TYPE_UNDEFINED
		#print("syntacticalNode.posTag = ", syntacticalNode.posTag)
		if(syntacticalNode.posTag in spacyPosTagTypeNoun):
			entityType = GIA_ENTITY_TYPE_SUBSTANCE
		elif(syntacticalNode.posTag in spacyPosTagTypeVerb):
			entityType = GIA_ENTITY_TYPE_ACTION
		elif(syntacticalNode.posTag in spacyPosTagTypePreposition):
			entityType = GIA_ENTITY_TYPE_CONDITION
		elif(syntacticalNode.posTag in spacyPosTagTypeAdverb):
			entityType = GIA_ENTITY_TYPE_QUALITY
		elif(syntacticalNode.posTag in spacyPosTagTypeAdjective):
			entityType = GIA_ENTITY_TYPE_QUALITY
		
		if(syntacticalNode.lemma == RELATION_ENTITY_HAVE):
			entityType = GIA_ENTITY_TYPE_PROPERTY
		elif(syntacticalNode.lemma == RELATION_ENTITY_BE):
			entityType = GIA_ENTITY_TYPE_DEFINITION
			
		#elif(syntacticalNode.posTag in spacyPosTagTypeConjunction):
		#	entityType = GIA_ENTITY_TYPE_CONDITION
		#elif(syntacticalNode.posTag in spacyPosTagTypeDeterminer):
		#	entityType =			
		#elif(sentenceLeafNodeList.posTag in spacyPosTagTypeQuestion):
		#	entityType =

		if(actionDetectionAnyCandidateVerbPOS):
			#override action dection using any candidate Verb POS 
			posValues = SPNLPpy_getAllPossiblePosTags.getAllPossiblePosTags(syntacticalNode.word)
			if(posValues in nltkPosTagTypeVerb):
				entityType = GIA_ENTITY_TYPE_ACTION
	else:
		entityType = syntacticalNode.entityType

	return entityType
	
