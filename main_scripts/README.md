# DeepPlant Models

This directory contains the scripts to train and evaluate the DeepPlant models. Training across different tasks is managed through configuration files located in the [`config/`](https://github.com/Addaoud/DeepPlant/tree/main/config) directory. 

> **Note:** The default parameters and hyperparameters in the provided `.json` files are the exact parameters used to generate the results reported and analyzed in our paper.

## 1. Setup & Requirements
If you are trying to replicate the results in the paper, you must first download our processed data. You can find it on [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19616358.svg)](https://doi.org/10.5281/zenodo.19616358)
Make sure to place the unzipped folder in the root `DeepPlant` directory. You can do this quickly via the command line:

```bash
# Navigate to the main DeepPlant directory
wget "https://zenodo.org/records/19616358/files/data.zip?download=1"
unzip data.zip
```

### Download Pre-trained Models (Optional)
If you wish to use our pre-trained models rather than training from scratch, download them from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19616358.svg)](https://doi.org/10.5281/zenodo.19616358) into the root directory:
```bash
# Navigate to the main DeepPlant directory
wget "https://zenodo.org/records/19616358/files/models.zip?download=1"
unzip models.zip
```

## 2. Usage & Command Line Interface
If you want to train and evaluate our models from scratch, you can use the scripts provided in this directory. All the main*.py scripts share a similar help guide and accept the following arguments:

```bash
usage: *main*.py [-h] [--json JSON] [-n] [-m MODEL] [-t] [-e]

Train and evaluate the DeepPlant models on chromatin state

options:
  -h, --help                Show the help message and exit.
  --json json_path          Required Path to the configuration .json file.
  -n, --new                 Build a new DeepPlant model from scratch.
  -m model_path, --model model_path
                            Load the weights of an existing DeepPlant model from the specified model_path.
  -t, --train               Train the model.
  -e, --evaluate            Evaluate the model.
```

## 3. Examples

### Chromatin State
To train and evaluate models on chromatin state, use the [main.py](https://github.com/Addaoud/DeepPlant/blob/main/main_scripts/main.py) script.

*Example: Train and evaluate a new model on the Arabidopsis data:*

```bash
python main.py --json "PATH_TO_JSON_FILE" -nte
```

### Downstream Tasks (Gene Expression)
To train and evaluate models on downstream tasks, use the main_expressions.py script.

*Example: Fine-tune a model using a pre-trained CSP model:*

```bash
python main_expressions.py --json "PATH_TO_JSON_FILE" -m "PATH_TO_CSP_MODEL" -nte
```

> **Note:** Using both `-m "PATH_TO_CSP_MODEL"` and `-n` together tells the script to load the pre-trained weights of the CSP model (excluding the final layer), build a new final layer, and fine-tune the entire model.

### Downstream Tasks (Enhancer Activity)
We provide two separate scripts for enhancer activity predictions:
1. [main_enhancer.py](https://github.com/Addaoud/DeepPlant/blob/main/main_scripts/main_enhancer.py): Uses DeepPlant embeddings to predict enhancer activity.
2. [main_enhancer_output.py](https://github.com/Addaoud/DeepPlant/blob/main/main_scripts/main_enhancer_output.py): Uses DeepPlant outputs to predict enhancer activity.

Refer to the table below for the specific scripts and configurations needed to train models for each task and plant species depending on the available data.

| Task     |Script               |Arabidopsis thaliana | Oryza sativa        | Joint (AT+OS)       |Zea mays             |
|----------|---------------------|---------------------|---------------------|---------------------|---------------------|
| CSP      |[main.py](https://github.com/Addaoud/DeepPlant/blob/main/main_scripts/main.py) | [config/config_AT_CSP.json](https://github.com/Addaoud/DeepPlant/blob/main/config/config_AT_CSP.json)|[config/config_AT_CSP.json](https://github.com/Addaoud/DeepPlant/blob/main/config/config_OS_CSP.json)|[config/config_AT_OS_CSP.json](https://github.com/Addaoud/DeepPlant/blob/main/config/config_AT_OS_CSP.json)|---------------------|
| GEP      |[main_expression.py](https://github.com/Addaoud/DeepPlant/blob/main/main_scripts/main_expression.py) | [config/config_AT_GEP.json](https://github.com/Addaoud/DeepPlant/blob/main/config/config_AT_GEP.json)|[config/config_OS_GEP.json](https://github.com/Addaoud/DeepPlant/blob/main/config/config_OS_GEP.json)|---------------------|---------------------|
| EAP      |[main_enhancer.py](https://github.com/Addaoud/DeepPlant/blob/main/main_scripts/main_enhancer.py) | [config/config_AT_EAP.json](https://github.com/Addaoud/DeepPlant/blob/main/config/config_AT_EAP.json)|---------------------|---------------------|[config/config_ZM_EAP.json](https://github.com/Addaoud/DeepPlant/blob/main/config/config_ZM_EAP.json)|

> **Note:** When using `main_enhancer_output.py`, you must configure the `targets` attribute in `config/config_AT_CSPtoEAP.json`. This determines which DeepPlant output factors are used to predict enhancer activity. Valid options are: `ALL`, `TF`, or `HM`.