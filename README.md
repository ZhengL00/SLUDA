# SLUDA

This repository contains the source code for our paper. 

## Setup
- **Build environment**
```
cd SLUDA
# use anaconda to build environment 
conda create -n STPar python=3.10
conda activate SLUDA
# install packages
pip install transformers==4.11.3
pip install spacy==3.4.1
pip install numpy==1.23.3
pip install spacy-transformers
python -m spacy download en_core_web_sm
```

## Quick Start

```
python main.py
```
