# DeepPlant

<p align="center">
  <img src="https://github.com/Addaoud/DeepPlant/tree/main/DeepPlant.PNG" width="640" alt="DeepPlant Architecture">
</p>

## Introduction
**DEEP-PLANT** is a sequence-based supervised foundation model for plant regulatory genomics. It is pre-trained on extensive chromatin state datasets from *Arabidopsis* and rice, and fine-tuned to predict downstream tasks such as gene expression and enhancer activity.

This repository contains the official codebase to reproduce the analyses performed in the DeepPlant paper, including data processing, model training, and evaluation across various genomic tasks.

**Try it out:** We have developed resources to assist users in predicting other genomic modalities from ATAC-seq without writing code:
*  **[Google Colab Notebook](#)**
*  **[DeepPlant Web Portal](#)**

## Setup & Installation
### 1. Clone the Repository
You can clone this repository using the command:
```bash
git clone https://github.com/Addaoud/DeepPlant.git
```

### 2. Create the Environment & Install Dependencies
We recommend using Conda to manage your environment. Run the following commands to create the environment, install the required packages, and set up your local paths:
```bash
cd DeepPlant
conda create -n DeepPlant python==3.11
conda activate DeepPlant
pip install -r requirements.txt
pip install -e
python3 setup_env.py
```
> **PyTorch Note:** If you encounter issues with PyTorch, ensure you install a version compatible with your specific CUDA version. Visit the [PyTorch Get Started](https://pytorch.org/) page for the latest instructions, or check the [Previous Versions](https://pytorch.org/get-started/previous-versions/) archive.

## Data & Pre-trained Models
### 1. Download the Training Data
The training data for DeepPlant (including metadata, dataset splits, `.fasta` sequence files, `.h5` files for CSP/GEP, and `.csv` files for EAP) is available via [Google Drive](https://drive.google.com/file/d/1_BmKIF9h9YqxynJUldfHd7e9Fox8iqn6/view?usp=drive_link) or [Zenodo](https://zenodo.org/).

Place the unzipped data directly in the root `DeepPlant` directory:

```bash
# Navigate to the main DeepPlant directory
gdown https://drive.google.com/file/d/1_BmKIF9h9YqxynJUldfHd7e9Fox8iqn6/view?usp=drive_link
unzip data.zip
```

### 2. Download Pre-trained Models
You can download all DeepPlant models trained on chromatin state, gene expression, and enhancer activity for Arabidopsis thaliana, Oryza sativa, and Zea mays from this [Google Drive](https://drive.google.com/drive/folders/1SvfHva4ll2ueiyWM6tS5xnJWB27Wg7O9?usp=sharing) Folder.

Place the unzipped models directly in the root `DeepPlant` directory:

```bash
# Navigate to the main DeepPlant directory
gdown https://drive.google.com/file/d/11pZKSEHv0ECaSp-l_BAQwoPKfeHxt9rR/view?usp=sharing
unzip models.zip
```

## Usage & Tutorials
We provide several Jupyter Notebooks to help you get started with DeepPlant's capabilities:
* [usage.ipynb](https://github.com/Addaoud/DeepPlant/blob/main/usage.ipynb): A comprehensive tutorial introducing how to use DeepPlant to predict multiple genomic modalities.
* [do_ISM_on_gene.ipynb](https://github.com/Addaoud/DeepPlant/blob/main/do_ISM_on_gene.ipynb): A notebook demonstrating how to identify positions and potential regulators of all 32,201 Arabidopsis genes using a comprehensive JASPAR database.

**In-Silico Mutagenesis (ISM) Analysis**

DeepPlant accurately identifies the regulators (upregulators and suppressors) of Arabidopsis genes under normal and stress conditions (cold, heat, wounding, drought) using in-silico mutagenesis and the Arabidopsis DeepPlant expression model.

You can view the results of our analysis on the DREB1 (DREB1A, DREB1B, DREB1C) and RD29A genes across different conditions in the [analysis/ISM](https://github.com/Addaoud/DeepPlant/tree/main/analysis/ISM) directory.

## Replicating Paper Results
To exactly replicate the model training and evaluation results presented in the DeepPlant paper:
1. Ensure the data is downloaded and placed in the root directory.
2. Use the main model training scripts located in the [main_scripts](https://github.com/Addaoud/DeepPlant/tree/main/main_scripts) directory.
3. The exact model architectures, training parameters, and hyperparameters used in the paper are stored in the `.json` files inside the [config](https://github.com/Addaoud/DeepPlant/tree/main/config) directory.


## Contributing
Contributions to this repository are welcome! If you find any bugs, have suggestions for new features, or want to improve the existing code, please create an issue or submit a pull request. For direct inquiries, please email **Ahmed Daoud** (Ahmed.Daoud@colostate.edu).