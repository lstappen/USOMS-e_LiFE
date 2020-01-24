**USOMS-e_lingustic_feature_extractor** is a simple Python pipeline to reproduce the extracted linguistic features from the `ComParE2020_USOMS-e` challenge. We utilise and provide contextual word embeddings using a frozen (not fine-tuned) German Bidirectional Language Transformer (Bert).

# Installation

Python 3.6 is required. The recommended type of installation is through `pip` in a separate virtualenv. 

### Python Installation

Assuming Ubuntu OS, execute the following steps:

1. Make sure that pip is up-to-date by running:
```
python -m pip install --user --upgrade pip
```
2. Install virtualenv
```
python -m pip install --user virtualenv
```
3. Create a new virtualenv som-e
```
python -m venv som-e
```
4. Activate the created virtualenv
```
source som-e/bin/activate
```
5. Install the packages according to requirements.txt - if GPU support is wished, the corresponding CUDA Toolkit etc. has to be installed first
```
pip install -r requirements.txt
```
6. Use the provided code & data or clone from github and add the data. Due to github upload restriction, precalculated embeddings can be downloaded here
```
embeddings: https://megastore.uni-augsburg.de/get/F3WcLLKsm9/
pos_tagger: https://megastore.uni-augsburg.de/get/KAIwgLRysv/
```
7. Run all models using
```
python run_all_models.py
```

### Python Dependencies (installed automatically during setup)
These Python packages are installed automatically during setup by `pip install -r requirements.txt`, and are just listed for completeness.

tensorflow==1.15.0
pytorch_transformers==1.2.0
torch==1.2.0
numpy==1.16.1
nltk==3.4.5
pandas==0.24.1
matplotlib==3.0.2
scikit_learn==0.22.1

### Additional Comments for GPU Usage

The pipeline is designed so that the final features from the models, which are subsequently fed into the SVM, can be computed with a desktop computer in reasonable computing time. By default, T_DEVICE is switched to `cpu`, so if the Bert extraction is re-calculated, torch (pytorch_transformers) does not move the Tensors to the GPU. Any other parameter e.g. `gpu` activates GPU usage. 

If GPU support is available, Tensorflow/Keras will still use GPU(s) for the model training, otherwise falls back on CPU and the pipeline runs entirely on CPU.

We provide pre-computed contextual word embeddings extracted from a frozen German Bert (last layer only) and a pre-trained German part-of-speech (POS) tagger model for all partitions of `ComParE2020_USOMS-e` to save computation time. They can also automatically recalculated by just deleting the provided `.npy` feature files in the corresponding directories. In this case, GPU support is recommended.

The compatible CUDA libraries (CUDA Toolkit, cuDNN) are required to be available on the system path (which should be the case after a standard installation). 

# Description

## General
All models include a sequence of words to story vector mapping (e.g. by a global max pooling or a BiLSTM + Attention) followed by one 512-dim FF (relu) and one 512-dim FF (sigmoid) layer. The output of this final layer is treated as the feature input for the SVM evaluation. 
`frozen-bert-pos-fuse-rnnatt` enriches the word embeddings (feature) inputs by POS tag inputs. This additional input is fed in a trainable POS embedding layer and trained by an auxiliary loss.

## Pipeline Process 

`run_all_models.py` trains and scores 
- all models specified in `executable_models`
- for labels specified in `labels`

Input granularity: All hand transcripted words/sentences of one story.

1. loading config file
2. loading text data and labels from .csv
3. (optional) computing (if not exist) or loading POS tagging and add it to the input
4. (optional) computing (if not exist) bert embedding
5. loading precalculated Bert features
6. creating model
7. training model, keep the best
8. exporting results to experiment folder: ckpt, config, features, graphs, results, svm_results
9. storing acc/loss graph from keras history
10. storing model features
11. storing exported features using SVM
12. fusing all features and score the fused features using SVM after all models are finished

## Further
The pipeline can also use the published machine transcriptions. In order to do this, the parameter config['text_column'] = 'hand_transcription' has to be changed to 'machine_transcription'. Subsequently, the contextual Bert embeddings for machine are calculated and extracted in the next run.

config['verbose'] parameter can be set > 0 to activate more output (e.g. model architecture)

# Acknowledgement 
The folder ClassifierBasedGermanTagger includes the code to train a basic German part-of-speech tagger. The reproduced accuracy is 96.09% on the (public) German TIGER corpus with a 90/10 split. Code is released under Apache 2.0 licence and can be found here: https://github.com/ptnplanet/NLTK-Contributions/tree/master/ClassifierBasedGermanTagger. All credits go to the authors.