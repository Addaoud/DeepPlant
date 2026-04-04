# DeepPlant

## Introduction
This repository contains shell and python scripts, notebooks, commands, etc. related to the analyses performed in the DeepPlant paper, including data processing, model training, and evaluation on different tasks, and interesting results!

## Installation
You can clone this repository using the command:
```bash
git clone https://github.com/Addaoud/DeepPlant.git
```

## Dependencies
1) You can create a conda environment and install the dependencies using pip:
```bash
conda create -n DeepPlant python==3.11
conda activate DeepPlant
pip install -r requirements.txt
```
2) You need to install a pytorch version compatible with your cuda version. You can follow the steps in [here](https://pytorch.org/) to install the latest pytorch version or you can refer to [previous versions](https://pytorch.org/get-started/previous-versions/) to install an older pytorch version. 

## Data Availability
The training data for DeepPlant can be downloaded from the following URL: 

## DeepPlant Model Training & Evaluation
To replicate the results presented in the paper, you can run the main main scripts for model training in [main_scripts]()
Model architecture, config, training parameters are in [config]()
Main scripts for model training are in 

## Paper results


## Usage
### Download the pre-training model and downstream models
You can download all DeepPlant models trained on chromatin state, gene expression, and enhancer activity in Arabidopsis thaliana, Oryza sativa, and Zea mays from [Google Drive](https://drive.google.com/drive/folders/1SvfHva4ll2ueiyWM6tS5xnJWB27Wg7O9?usp=sharing)
For the trained downstream models and how to train downstream models from scratch, you can go to each correspoding directory

### Tutorials: ISM
DeepPlant has been shown to identify, with precision, the regulators (upregulators and suppressors) of Arabidopsis genes under normal conditions and in different stress conditions (cold, heat, wounding, drought) using in-silico-mutagenesis and the Arabidopsis DeepPlant expression model.
We provide results of our analysis on the DREB1 (DREB1A, DREB1B, DREB1C) and the RD29A genes in different conditions in [ISM]().

We also provide a notebook [do_ISM_on_gene.ipynb] to identify the positions and the potential regulators of all the different 32201 Arabidopsis genes included in our gene expression data using a comprehensive Jaspar database and 


## Contributing
Contributions to this repository are welcome! If you find any bugs, have suggestions for new features, or want to improve the existing code, please create an issue or submit a pull request. You can post in the Github issues or e-mail Ahmed Daoud (Ahmed.Daoud@colostate.edu) or Asa Ben-Hur (Asa.Ben-Hur@colostate.edu).