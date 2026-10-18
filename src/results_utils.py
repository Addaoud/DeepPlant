import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn import metrics
from scipy.stats import pearsonr, spearmanr
from typing import List, Tuple, Optional
from .utils import save_data_to_csv
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


def _gather_eval(records, preds, targets):
    """Gather (records, preds, targets) from all ranks and drop the duplicates
    added by DistributedSampler padding. All ranks must call this."""
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, (records, preds.cpu(), targets.cpu()))
    recs = [r for o in out for r in o[0]]
    P = torch.cat([o[1] for o in out])
    T = torch.cat([o[2] for o in out])
    _, keep = np.unique(recs, return_index=True)
    keep = np.sort(keep)
    return [recs[i] for i in keep], P[keep], T[keep]


def evaluate_model(
    model,
    dataloader,
    device,
    activation_function=None,
    results_folder_path="",
    experiment_names=[],
    per_gene: bool = False,
    distributed: bool = False,  # True only if ALL ranks call evaluate_model
):
    logger = configure_logging_format(file_path=results_folder_path)
    logger.info(f"Device: {device} - Evaluating the model (per_gene={per_gene})")
    target_dict, preds_dict, record_dict = dict(), dict(), dict()
    if activation_function is None:
        activation_function = torch.nn.Identity()

    with torch.no_grad():
        model.eval()
        for (record, data, target), bit in progress_bar(dataloader):
            data = {k: v.to(device) for k, v in data.items()}
            output = activation_function(model(**data, bit=bit))
            if isinstance(output, tuple):
                output = output[0]
            if output.ndim == 1:
                output = output.unsqueeze(0)
            if target.ndim == 3:
                target = target.reshape(-1, target.shape[2])
            preds_dict.setdefault(bit, []).append(output.float().cpu())
            target_dict.setdefault(bit, []).append(target.float().cpu())
            record_dict.setdefault(bit, []).extend(list(record))

    for key in target_dict:
        records_full = record_dict[key]
        preds_full = torch.cat(preds_dict[key], dim=0)
        target_full = torch.cat(target_dict[key], dim=0)
        if distributed:
            records_full, preds_full, target_full = _gather_eval(
                records_full, preds_full, target_full
            )
        if not is_main_process():
            continue

        preds_np = preds_full.numpy().astype(np.float32)
        target_np = target_full.numpy().astype(np.float32)
        logger.info(
            f"Evaluating {len(records_full)} records, "
            f"{preds_np.shape[1]} targets (bit={key})"
        )

        # ---- per-track (always) ----
        pearsonr_list, spearmanr_list, mse_list = [], [], []
        for i in range(preds_np.shape[1]):
            pcc, _ = pearsonr(preds_np[:, i], target_np[:, i])
            spm, _ = spearmanr(preds_np[:, i], target_np[:, i])
            pearsonr_list.append(pcc)
            spearmanr_list.append(spm)
            mse_list.append(metrics.mean_squared_error(preds_np[:, i], target_np[:, i]))

        mean_pearsonr = np.nanmean(pearsonr_list)
        mean_spearmanr = np.nanmean(spearmanr_list)
        mean_mse = np.nanmean(mse_list)

        if results_folder_path:
            plot_distribution(
                list=pearsonr_list,
                file_path=os.path.join(results_folder_path, f"distribution{key}.png"),
            )
            try:
                target_list = load_targets(targets_file_path=experiment_names[key])
            except Exception:
                target_list = np.arange(len(pearsonr_list))
            pd.DataFrame(
                {
                    "Experiment ID": np.arange(len(target_list)),
                    "Target": target_list,
                    "Pearsonr correlation": pearsonr_list,
                    "Spearman correlation": spearmanr_list,
                    "MSE": mse_list,
                }
            ).sort_values(by="Pearsonr correlation", ascending=False).to_csv(
                os.path.join(results_folder_path, f"Correlation_df{key}.csv"),
                index=False,
            )

            if per_gene:
                g_pcc, g_spm, g_mse = [], [], []
                for r in range(preds_np.shape[0]):
                    pcc, _ = pearsonr(preds_np[r, :], target_np[r, :])
                    spm, _ = spearmanr(preds_np[r, :], target_np[r, :])
                    g_pcc.append(pcc)
                    g_spm.append(spm)
                    g_mse.append(
                        metrics.mean_squared_error(preds_np[r, :], target_np[r, :])
                    )
                pd.DataFrame(
                    {
                        "Gene index": np.arange(len(g_pcc)),
                        "Gene": records_full,
                        "Pearsonr correlation": g_pcc,
                        "Spearman correlation": g_spm,
                        "MSE": g_mse,
                    }
                ).sort_values(by="Pearsonr correlation", ascending=False).to_csv(
                    os.path.join(
                        results_folder_path, f"Correlation_perGene_df{key}.csv"
                    ),
                    index=False,
                )

        yield (float(mean_mse), float(mean_pearsonr), float(mean_spearmanr))


def plot_distribution(
    list: List[float], file_path: str, use_log_scale: Optional[bool] = False
):
    sns.displot(list)
    plt.xlabel("Value")
    if use_log_scale:
        plt.yscale("log")
    plt.savefig(file_path)
    plt.close()


def _to_scores(P: torch.Tensor) -> torch.Tensor:
    """(N,), (N,1) or (N,2) model outputs -> P(class 1).
    Applies softmax/sigmoid if the outputs are logits rather than probabilities."""
    if P.ndim == 2 and P.shape[1] == 2:
        if not torch.allclose(P.sum(1), torch.ones(len(P)), atol=1e-4):
            P = torch.softmax(P, dim=1)
        return P[:, 1]
    P = P.reshape(-1)
    if P.min() < 0 or P.max() > 1:
        P = torch.sigmoid(P)
    return P


def evaluate_model_classification(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    activation_function=None,
    results_folder_path: Optional[str] = "",
    distributed: bool = False,  # True only if ALL ranks call this function
) -> tuple[float, float, float]:
    """Evaluate on the full dataloader; return (accuracy, auroc, auprc)."""
    logger = configure_logging_format(file_path=results_folder_path)
    logger.info(f"Device: {device} - Evaluating the model")
    if activation_function is None:
        activation_function = torch.nn.Identity()

    preds, targets, records = [], [], []
    with torch.no_grad():
        model.eval()
        for (header, data, target), bit in progress_bar(dataloader):
            data = {k: v.to(device) for k, v in data.items()}
            output = activation_function(model(**data, bit=bit))
            if isinstance(output, tuple):
                output = output[0]
            preds.append(output.float().cpu())
            targets.append(target.reshape(-1).cpu())
            records.extend(header.tolist() if torch.is_tensor(header) else list(header))

    P, T = torch.cat(preds), torch.cat(targets)
    if distributed:
        records, P, T = _gather_eval(records, P, T)

    result = None
    if is_main_process():
        scores, T = _to_scores(P), T.long()
        auroc = AUROC(task="binary")(scores, T).item()
        auprc = AveragePrecision(task="binary")(scores, T).item()
        accuracy = ((scores >= 0.5).long() == T).float().mean().item()
        logger.info(
            f"Evaluated {len(records)} sequences "
            f"({int(T.sum())} positives): AUROC={auroc:.4f}, AUPRC={auprc:.4f}"
        )
        if results_folder_path:
            pd.DataFrame(
                {"seq_idx": records, "label": T.tolist(), "prediction": scores.tolist()}
            ).to_csv(os.path.join(results_folder_path, "predictions.csv"), index=False)
        result = (accuracy, auroc, auprc)

    if distributed:  # every rank gets the same metrics (e.g., for early stopping)
        obj = [result]
        dist.broadcast_object_list(obj, src=0)
        result = obj[0]
    return result
