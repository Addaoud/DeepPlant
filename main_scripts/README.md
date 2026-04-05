# DeepPlant Models

This directory contains the scripts to train and evaluate the DeepPlant models. Training across different tasks is managed through configuration files located in the [`config/`](https://github.com/Addaoud/DeepPlant/tree/main/config) directory. 

> **Note:** The default parameters and hyperparameters in the provided `.json` files are the exact parameters used to generate the results reported and analyzed in our paper.

## 1. Setup & Requirements
If you are trying to replicate the results in the paper, you must first download our processed data. You can find it on [Google Drive](https://drive.google.com/file/d/1_BmKIF9h9YqxynJUldfHd7e9Fox8iqn6/view?usp=drive_link) or [Zenodo](https://zenodo.org/). 

Make sure to place the unzipped folder in the root `DeepPlant` directory. You can do this quickly via the command line:

```bash
# Navigate to the main DeepPlant directory
gdown [https://drive.google.com/file/d/1_BmKIF9h9YqxynJUldfHd7e9Fox8iqn6/view?usp=drive_link](https://drive.google.com/file/d/1_BmKIF9h9YqxynJUldfHd7e9Fox8iqn6/view?usp=drive_link)
unzip data.zip
```

### Download Pre-trained Models (Optional)
If you wish to use our pre-trained models rather than training from scratch, download them from [Google Drive](https://drive.google.com/file/d/11pZKSEHv0ECaSp-l_BAQwoPKfeHxt9rR/view?usp=sharing) into the root directory:
```bash
gdown [https://drive.google.com/file/d/11pZKSEHv0ECaSp-l_BAQwoPKfeHxt9rR/view?usp=sharing](https://drive.google.com/file/d/11pZKSEHv0ECaSp-l_BAQwoPKfeHxt9rR/view?usp=sharing)
unzip models.zip
```

## 2. Usage & Command Line Interface
If you want to train and evaluate our models from scratch, you can use the scripts provided in this directory. All the main*.py scripts share a similar help guide and accept the following arguments:

```bash
usage: *script*.py [-h] [--json JSON] [-n] [-m MODEL] [-t] [-e]

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

> **Note:** Using both '-m "PATH_TO_CSP_MODEL"' and '-n' together tells the script to load the pre-trained weights of the CSP model (excluding the final layer), build a new final layer, and fine-tune the entire model.

### Downstream Tasks (Enhancer Activity)
We provide two separate scripts for enhancer activity predictions:
    1. [main_enhancer.py](https://github.com/Addaoud/DeepPlant/blob/main/main_scripts/main_enhancer.py): Uses DeepPlant embeddings to predict enhancer activity.
    2. [main_enhancer_output.py](https://github.com/Addaoud/DeepPlant/blob/main/main_scripts/main_enhancer_output.py): Uses DeepPlant outputs to predict enhancer activity.

This table summarizes the scripts and config files used to train the different task models on different plant species

| Task     |Script               |Arabidopsis thaliana | Oryza sativa        | Zea mays            |
|----------|---------------------|---------------------|---------------------|---------------------|
| CSP      |[main.py](https://github.com/Addaoud/DeepPlant/blob/main/main_scripts/main.py) | [config/config_AT_2500.json](https://github.com/Addaoud/DeepPlant/blob/main/config/config_AT_2500.json)|[config/config_AT_2500.json](https://github.com/Addaoud/DeepPlant/blob/main/config/config_OS_2500.json)| --- |
|----------|---------------------|---------------------|---------------------|---------------------|
| GEP      |[main_expression.py](https://github.com/Addaoud/DeepPlant/blob/main/main_scripts/main_expression.py) | [config/config_AT_expression.json](https://github.com/Addaoud/DeepPlant/blob/main/config/config_AT_expression.json)|[config/config_AT_2500.json](https://github.com/Addaoud/DeepPlant/blob/main/config/config_OS_2500.json)| --- |

