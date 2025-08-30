import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn import metrics
from scipy.stats import pearsonr, spearmanr
from typing import List, Tuple, Optional
from fastprogress import progress_bar
from .seed import set_seed
from .ddp import is_dist_avail_and_initialized, is_main_process
from src.logger import configure_logging_format
import os
import pandas as pd
import torch.distributed as dist
from torchmetrics import AUROC, AveragePrecision
import warnings

warnings.filterwarnings("ignore")

sns.set_theme()
set_seed()


def load_targets(targets_file_path: str) -> pd.DataFrame:
    targets = open(targets_file_path, "r").read().splitlines()
    return targets


def onehot_encode_labels(labels_list: List[int], num_labels: int) -> np.array:
    """
    creates one hot encoded labels, e.g. [0,1,2] => [ [1,0,0] , [0,1,0] , [0,0,1] ]
    :param labels_list:
        list containing the labels
    :param num_labels:
        number of unique labels
    :return:
        numpy array containing the one hot encoded labels list
    """
    return np.eye(num_labels)[labels_list]


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.dataloader,
    device: str,
    activation_function: torch.nn.modules.loss = None,
    model_folder_path: Optional[str] = "",
    experiment_names: Optional[List[str]] = [],
) -> Tuple[float, float, float]:
    """
    evaluate the model on dataloader and return the mse, pearsonR, and SpearmanR
    """
    logger = configure_logging_format(file_path=model_folder_path)
    logger.info(f"Device: {device} - Evaluating the model")
    target_dict = dict()
    preds_dict = dict()
    if activation_function == None:
        activation_function = torch.nn.Identity()
    with torch.no_grad():
        model.eval()
        for idx, ((_, data, target), bit) in enumerate(progress_bar(dataloader)):
            data = {key: data[key].to(device) for key in data}
            output = activation_function(model(**data, bit=bit))
            if type(output) == tuple:
                output = output[0]
            preds = output.cpu().detach().numpy()
            if target.ndim == 3:
                target = target.view(target.shape[0] * target.shape[1], target.shape[2])
            target_dict[bit] = target_dict.get(bit, list()) + target.tolist()
            preds_dict[bit] = preds_dict.get(bit, list()) + preds.tolist()
    target_array_dict = {
        key: np.array(target_list) for key, target_list in target_dict.items()
    }
    preds_array_dict = {
        key: np.array(preds_list) for key, preds_list in preds_dict.items()
    }
    mean_pearsonr = torch.zeros(1).to(device)
    mean_spearmanr = torch.zeros(1).to(device)
    mean_mse = torch.zeros(1).to(device)
    for key, target_array in target_array_dict.items():
        mse_list = list()
        pearsonr_list = list()
        spearmanr_list = list()
        preds_array = preds_array_dict[key]
        for i in range(target_array.shape[1]):
            pcc, _ = pearsonr(preds_array[:, i], target_array[:, i])
            spm, _ = spearmanr(preds_array[:, i], target_array[:, i])
            mse = metrics.mean_squared_error(preds_array[:, i], target_array[:, i])
            pearsonr_list.append(pcc)
            spearmanr_list.append(spm)
            mse_list.append(mse)
        pearson_corr = np.nanmean(pearsonr_list)
        mean_pearsonr += pearson_corr
        spearman_corr = np.nanmean(spearmanr_list)
        mean_spearmanr += spearman_corr
        mse = np.nanmean(mse_list)
        mean_mse += mse
        if is_dist_avail_and_initialized():
            pearsonr_list = torch.tensor(
                pearsonr_list, dtype=torch.float32, device=device
            )
            spearmanr_list = torch.tensor(
                spearmanr_list, dtype=torch.float32, device=device
            )
            mse_list = torch.tensor(mse_list, dtype=torch.float32, device=device)
            dist.all_reduce(mean_pearsonr, dist.ReduceOp.SUM)
            dist.all_reduce(mean_spearmanr, dist.ReduceOp.SUM)
            dist.all_reduce(mean_mse, dist.ReduceOp.SUM)
            dist.all_reduce(pearsonr_list, dist.ReduceOp.SUM)
            dist.all_reduce(spearmanr_list, dist.ReduceOp.SUM)
            dist.all_reduce(mse_list, dist.ReduceOp.SUM)
            mean_pearsonr /= dist.get_world_size()
            mean_spearmanr /= dist.get_world_size()
            mean_mse /= dist.get_world_size()
            pearsonr_list = (pearsonr_list / dist.get_world_size()).tolist()
            spearmanr_list = (spearmanr_list / dist.get_world_size()).tolist()
            mse_list = (mse_list / dist.get_world_size()).tolist()
        if model_folder_path and is_main_process():
            plot_distribution(
                list=pearsonr_list,
                file_path=os.path.join(model_folder_path, f"distribution{key}.png"),
            )
            try:
                target_list = load_targets(targets_file_path=experiment_names[key])
            except:
                target_list = np.arange(0, len(pearsonr_list))
            Correlation_csv_file = os.path.join(
                model_folder_path, f"Correlation_df{key}.csv"
            )
            experiment_id = np.arange(0, len(target_list))
            df = pd.DataFrame(
                {
                    "Experiment ID": experiment_id,
                    "Target": target_list,
                    "Pearsonr correlation": pearsonr_list,
                    "Spearman correlation": spearmanr_list,
                    "MSE": mse_list,
                }
            ).sort_values(by="Pearsonr correlation", ascending=False)
            df.to_csv(Correlation_csv_file, index=False)
    return (
        mean_mse.item() / len(target_array_dict),
        pearson_corr.item() / len(target_array_dict),
        spearman_corr.item() / len(target_array_dict),
    )


def plot_distribution(
    list: List[float], file_path: str, use_log_scale: Optional[bool] = False
):
    sns.displot(list)
    plt.xlabel("Value")
    if use_log_scale:
        plt.yscale("log")
    plt.savefig(file_path)
    plt.close()


def evaluate_model_classification(
    model: torch.nn.Module,
    dataloader: torch.utils.data.dataloader,
    device: str,
    activation_function: torch.nn.modules.loss = None,
    model_folder_path: Optional[str] = "",
) -> tuple[float, float, float]:
    """
    evaluate the model on dataloader and return the accuracy, auroc, and auprc
    """
    logger = configure_logging_format(file_path=model_folder_path)
    logger.info(f"Device: {device} - Evaluating the model")
    target_list = list()
    preds_list = list()
    auroc_fn = AUROC(task="binary")
    aupr_fn = AveragePrecision(task="binary")
    if activation_function == None:
        activation_function = torch.nn.Identity()
    with torch.no_grad():
        model.eval()
        for _, ((header, data, target), bit) in enumerate(progress_bar(dataloader)):
            data = {key: data[key].to(device) for key in data}
            output = activation_function(model(**data, bit=bit))
            preds = output.cpu().detach().numpy()
            target_list.extend(target)
            preds_list.extend(preds)
    if preds.shape[1] == 1:
        auroc = auroc_fn(
            torch.from_numpy(np.array(preds_list)),
            torch.from_numpy(np.array(target_list)),
        ).to(device)

        auprc = aupr_fn(
            torch.from_numpy(np.array(preds_list)),
            torch.from_numpy(np.array(target_list)),
        ).to(device)
        amax = np.round(preds_list) == target_list
        accuracy = sum(list(amax)).item() / len(target_list)
    elif preds.shape[1] == 2:
        auroc = auroc_fn(
            torch.from_numpy(np.array(preds_list)[:, 1]),
            torch.from_numpy(np.array(target_list)),
        ).to(device)

        auprc = aupr_fn(
            torch.from_numpy(np.array(preds_list)[:, 1]),
            torch.from_numpy(np.array(target_list)),
        ).to(device)
        amax = np.round(np.array(preds_list)[:, 1]) == target_list
        accuracy = torch.tensor(sum(list(amax)).item() / len(target_list)).to(device)
    if is_dist_avail_and_initialized():
        dist.all_reduce(auroc, dist.ReduceOp.SUM)
        dist.all_reduce(auprc, dist.ReduceOp.SUM)
        dist.all_reduce(accuracy, dist.ReduceOp.SUM)
        auroc /= dist.get_world_size()
        auprc /= dist.get_world_size()
        accuracy /= dist.get_world_size()
    return (accuracy.item(), auroc.item(), auprc.item())
