# DeepPlant

<p align="center">
  <img src="https://github.com/Addaoud/DeepPlant/blob/main/DeepPlant.jpg" width="640" alt="DeepPlant Architecture">
</p>

## Introduction
**DEEP-PLANT** is a sequence-based supervised foundation model for plant regulatory genomics. It is pre-trained on extensive chromatin state datasets from *Arabidopsis* and rice, and fine-tuned to predict downstream tasks such as gene expression and enhancer activity.

This repository contains the official codebase to reproduce the analyses performed in the DeepPlant paper, including data processing, model training, and evaluation across various plant genomic tasks.

## Manuscript
Daoud Ahmed, Soumyadip Roy, Haoxuan Zeng, Xinyu Bao, Zhenhao Zhang, Jiakang Wang, Paul Parodi, Anireddy Reddy, Jie Liu, and Asa Ben-Hur. "Deep-Plant: a supervised foundation model for plant regulatory genomics." bioRxiv (2026): 2026-04.

## System Requirements

### Hardware requirements
DeepPlant runs on a standard computer with sufficient RAM to handle model operations. While a GPU can significantly speed up computations, it is not required.

### Software requirements
This package is supported for macOS and Linux. The package has been tested on the following systems:
 * macOS: Ventura (13.7.8)
 * Linux: AlmaLinux 9.7
 * CUDA Version: 13.0


## Setup & Installation
### 1. Clone the Repository
You can clone this repository using the command:
```bash
git clone https://github.com/Addaoud/DeepPlant.git
```

### 2. Create the Environment & Install Dependencies
We recommend using [Conda](https://www.anaconda.com/download) to manage your environment. This repository requires python 3.11+ and further depends on a number of python packages (which are found in [requirements.txt](https://github.com/Addaoud/DeepPlant/blob/main/requirements.txt) and are automatically installed with the conda eenvironment using [environment.yml](https://github.com/Addaoud/DeepPlant/blob/main/environment.yml)). Run the following commands to create the conda environment, install the required packages, and set up your local paths:
```bash
cd DeepPlant
conda env create -f environment.yml
conda activate deepplant
python setup_env.py
```
> **PyTorch Note:** If you encounter issues with PyTorch, ensure you install a version compatible with your specific CUDA version. Visit the [PyTorch Get Started](https://pytorch.org/) page for the latest instructions, or check the [Previous Versions](https://pytorch.org/get-started/previous-versions/) archive.

## Data & Pre-trained Models
### 1. Download the Training Data
The training data for DeepPlant (including metadata, dataset splits, `.fasta` sequence files, `.h5` files for CSP/GEP, and `.csv` files for EAP) is available via [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19616358.svg)](https://doi.org/10.5281/zenodo.19616358). The dataset is approximately 26.5 GB in size and may take some time to download, depending on internet speed.

Place the unzipped data directly in the root `DeepPlant` directory:

```bash
# Navigate to the main DeepPlant directory
wget "https://zenodo.org/records/19616358/files/data.zip?download=1"
unzip data.zip
```

### 2. Download Pre-trained Models
You can download all DeepPlant models trained on chromatin state, gene expression, and enhancer activity for Arabidopsis thaliana, Oryza sativa, and Zea mays from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19616358.svg)](https://doi.org/10.5281/zenodo.19616358). The models are approximately 9.9 GB in size and may take some time to download, depending on internet speed.

Place the unzipped models directly in the root `DeepPlant` directory:

```bash
# Navigate to the main DeepPlant directory
wget "https://zenodo.org/records/19616358/files/models.zip?download=1"
unzip models.zip
```

## Usage & Tutorials
Before using DeepPlant, activate the conda environment:
```bash
conda activate deepplant
```
We provide both a Python script and Jupyter notebooks to help you get started.

Make sure to download and extract the *models/* and *data/* directories before running any utilities.

### Input Specification
DeepPlant supports three predictive tasks:
* CSP — Chromatin State Prediction
* GEP — Gene Expression Prediction
* EAP — Enhancer Activity Prediction

Supported species:
* arabidopsis - *Arabidopsis thaliana*
* rice - *Oryza sativa*

Each prediction requires exactly one of the following inputs:
* FASTA sequence (--fasta)
* Genomic locus (e.g., Chr1:10000-12500)
* Gene identifier (recommended for GEP; e.g., AT5G52310)

### Sequence Processing
DeepPlant operates on a fixed 2.5 kb input sequence centered on the region of interest:
* Sequences > 2.5 kb are center-cropped
* Sequences < 2.5 kb are padded with N's

Predictions are computed over the central region of the sequence:
* CSP: epigenomic features across the central window (200 bps)
* EAP: enhancer probability within the central ~200 bp

> **Limitation:**  Enhancer Activity Prediction (*EAP*) is not supported for rice due to the lack of a high-quality enhancer dataset.

### 1. Prediction Script
DeepPlant provides a unified interface via:
* [DeepPlant.py](https://github.com/Addaoud/DeepPlant/blob/main/DeepPlant.py): A command-line tool for predicting multiple genomic modalities.
```bash
usage: DeepPlant.py [-h] --species {arabidopsis,rice} --task {CSP,GEP,EAP} --output OUTPUT (--fasta FASTA | --locus LOCUS | --gene GENE)

DeepPlant Prediction Script

options:
  -h, --help            show this help message and exit
  --species {arabidopsis,rice}
  --task {CSP,GEP,EAP}
  --output OUTPUT       Path to output folder
  --fasta FASTA         Path to fasta file
  --locus LOCUS         Locus string e.g. Chr1:1000-3500
  --gene GENE           Gene name e.g. AT5G52310
```

#### Example: Chromatin State Prediction (CSP)
We provide a demo fasta file [Dema/seq.fa](https://github.com/Addaoud/DeepPlant/blob/main/Demo/seq.fa) that contains the header *myseq* and a DNA sequence of length 2.5kb. 

Predict chromatin state for a 2.5 kb sequence:
```bash
python3 DeepPlant.py --species "arabidopsis" --task "CSP" --output "Demo" --fasta "Demo/seq.fa"
```
Output:
```
Loading model state
Model state loaded
Results saved to: Demo/sequence_CSP_results.csv
--- Process Complete ---
Species: arabidopsis | Task: CSP
```
You can also use a genomic locus:
```bash
python3 DeepPlant.py --species "rice" --task "CSP" --output "Demo" --locus "Chr1:8000-10500"
```
Output:
```
Loading model state
Model state loaded
Results saved to: Demo/rice_Chr1_8000_10500_CSP_results.csv
--- Process Complete ---
Species: rice | Task: CSP
```
#### Example: Gene Expression Prediction (GEP)
Gene-based prediction is recommended:
 ```bash
python3 DeepPlant.py --species "arabidopsis" --task "GEP" --output "Demo" --gene "AT5G52310"
```
Output:
```
Loading model state
Model state loaded
Results saved to: Demo/AT5G52310_5_21240717_GEP_results.csv
--- Process Complete ---
Species: arabidopsis | Task: GEP
```
The output CSV contains predicted expression across tissues and conditions.

#### Example: Enhancer Activity Prediction (EAP)
```bash
python3 DeepPlant.py --species "arabidopsis" --task "EAP" --output "Demo" --fasta "Demo/seq.fa
```
Output:
```
Loading model state
Model state loaded
The probability of the presence of an enhancer in the center of the given DNA sequence is 99.996%
--- Process Complete ---
Species: arabidopsis | Task: EAP
```
You can also use a genomic locus (e.g., *Chr3:151000-153500*).

### 2. In-Silico Mutagenesis (ISM) Analysis
[ISM.ipynb](https://github.com/Addaoud/DeepPlant/blob/main/ISM.ipynb) is An interactive notebook for identifying regulatory regions and transcription factor drivers using:
* In-silico mutagenesis (ISM)
* JASPAR motif database
* ChIP-seq evidence

DeepPlant uses ISM to infer activators and repressors of *Arabidopsis thaliana* genes under:
* Normal conditions
* Stress conditions (cold, heat, drought, wounding, etc...)

#### Example Results
You can view the analysis of DREB1 genes (DREB1A, DREB1B, DREB1C) here [analysis/ISM](https://github.com/Addaoud/DeepPlant/tree/main/analysis/ISM).

## Replicating Paper Results
To replicate the model training and evaluation results presented in the DeepPlant paper:
1. Ensure the data is downloaded and placed in the root directory.
2. Use the main model training scripts located in the [main_scripts](https://github.com/Addaoud/DeepPlant/tree/main/main_scripts) directory.
3. The exact model architectures, training parameters, and hyperparameters used in the paper are stored in the `.json` files inside the [config](https://github.com/Addaoud/DeepPlant/tree/main/config) directory.

## under constructions
**Try it out:** We are developing a Hugging Face web portal that allows you to view and compare DeepPlant's epigenomic predictions against real, genome-wide tracks. The tool automatically generates ready-to-use Integrative Genomics Viewer (IGV) links for seamless exploration.
*  **[DeepPlant Web Portal](https://huggingface.co/spaces/soumya160497/plant_genome_1)**

## Licence
This project is covered under the **Apache 2.0 License**.

## Contributing
Contributions to this repository are welcome! If you find any bugs, have suggestions for new features, or want to improve the existing code, please create an issue or submit a pull request. For direct inquiries, please email **Ahmed Daoud** (Ahmed.Daoud@colostate.edu).