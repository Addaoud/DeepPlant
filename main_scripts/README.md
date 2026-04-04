# DeepPlant models
This directory contains the scripts to train and evaluate the DeepPlant models. Training on the different tasks was made easier through a series of config files in [config](). The default parameters and hyperparameters in the json files are the parameters used for the models that we reported and analyzed the results for in the paper.

## Data
If you are trying to replicate the results in the paper, be sure to first download our processed data from one of the following links [drive] or [zenodo].


## Train and evaluate the DeepPlant models
All the models have the same guiding help (you can replace *script* by the appropriate python script name).
```bash
usage: *script*.py [-h] [--json JSON] [-n] [-m MODEL] [-t] [-e]

Train and evaluate the DeepPlant models on chromatin state

options:
  -h, --help            show this help message and exit
  --json JSON           Path to the json file
  -n, --new             Use this option to build a new DeepPlant model
  -m MODEL, --model MODEL
                        Use this option to load the weights of an existing DeepPlant model from model_path
  -t, --train           Use this option to train the model
  -e, --evaluate        Use this option to evaluate the model
```
    * [main.py]() is the script used to train and evaluate the models on chromatin state. For example, to train and evaluate the DeepPlant model on the arabidopsis data, use the following command
```bash
python main.py --json "config/config_AT_2500.json" -nte
```
    * [main_expressions]() is the script used to train and and evaluate the models on gene expression.
```bash
python main_expressions.py --json "config/config_AT_2500.json" -nte
```
    * [main_enhancer]() is the script used to train and and evaluate the models on enhancer activity.
```bash
python main_enhancer.py --json "config/config_AT_2500.json" -nte
```
    * [main_enhancer_output]() is the script used to train and and evaluate the models on enhancer activity using DeepPlant chromatin state features.
```bash
python main_enhancer_output.py --json "config/config_AT_2500.json" -nte
```



