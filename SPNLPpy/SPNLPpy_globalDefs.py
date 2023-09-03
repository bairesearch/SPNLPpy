"""SPNLPpy_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SPNLPpy_main.py

# Usage:
see SPNLPpy_main.py

# Description:
SPNLP - global defs

"""

useSPNLPcustomSyntacticalParser = False

if(useSPNLPcustomSyntacticalParser):
	constituencyParserType = "constituencyParserWordVector"	#default algorithmSPNLP:generateSyntacticalGraph	#experimental
else:
	constituencyParserType = "constituencyParserFormal"

if(useSPNLPcustomSyntacticalParser):
	dependencyParserType = "dependencyParserWordVector"	#default algorithmSPNLP:generateSyntacticalGraph	#experimental
else:
	dependencyParserType = "dependencyParserFormal"
	
if(dependencyParserType == "dependencyParserWordVector"):
	generateDependencyParseTreeFromConstituencyParseTree = False
elif(dependencyParserType == "dependencyParserFormal"):
	pass
	
drawSyntacticalGraph = False
if(drawSyntacticalGraph):	
	drawSyntacticalGraphSentence = True
	drawSyntacticalGraphNetwork	= True	#draw graph for entire network (not just sentence)
	drawSyntacticalGraphNodeColours = False	#enable for debugging SPNLPpy_syntacticalGraphIntermediaryTransformation
else:
	drawSyntacticalGraphSentence = False
	drawSyntacticalGraphNetwork = False
	drawSyntacticalGraphNodeColours = False
	
performReferenceResolution = True
