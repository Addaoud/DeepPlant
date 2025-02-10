import torch
import numpy as np
import random
import os


def set_seed(seed: int = 42) -> None:
    """
    Code was obtained from https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
    The torch.backends.cudnn.benchmark is set to true to improve running time
    Set Random Seeds and improve results reproducibility
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)
