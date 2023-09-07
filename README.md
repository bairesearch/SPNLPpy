# SPNLPpy

### Author

Richard Bruce Baxter - Copyright (c) 2022 Baxter AI (baxterai.com)

### Description

Syntactical Parser Natural Language Processing (SPNLP) for Python - experimental 

### License

MIT License

### Installation
```
conda create -n anntf2 python=3.7
source activate anntf2
pip install networkx [required for SPNLPpy_syntacticalGraphDraw]
pip install matplotlib==2.2.3 [required for SPNLPpy_syntacticalGraphDraw]
pip install nltk spacy==2.3.7
python3 -m spacy download en_core_web_md
pip install benepar [required for SPNLPpy_syntacticalGraphConstituencyParserFormal]
```

### Execution
```
source activate anntf2
python3 SPNLPpy_main.py
```
