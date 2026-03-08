import torch
import numpy as np
import os
from datetime import datetime
from .utils import save_data_to_csv, plot_loss, get_paths
from .seed import set_seed
from .ddp import is_main_process, is_dist_avail_and_initialized
from .logger import configure_logging_format
from torchmetrics import AveragePrecision
from fastprogress import progress_bar
from typing import Optional
import torch.distributed as dist
from copy import deepcopy
from datetime import datetime
from scipy.stats import pearsonr

set_seed()


def pearson(preds_array: np.array, target_array: np.array) -> torch.Tensor:
    if target_array.ndim == 3:
        target_array = target_array.reshape(-1, target_array.shape[2])
    pearsonr_list = list()
    for i in range(target_array.shape[1]):
        pcc, _ = pearsonr(preds_array[:, i], target_array[:, i])
        pearsonr_list.append(pcc)
    pearson_corr = torch.zeros(1)
    pearson_corr += np.nanmean(pearsonr_list)
    return pearson_corr


def auprc(preds_array: np.array, target_array: np.array) -> torch.Tensor:
    aupr_fn = AveragePrecision(task="binary")
    if preds_array.ndim == 1:
        auprc_value = aupr_fn(
            torch.from_numpy(preds_array),
            torch.from_numpy(target_array),
        )
    elif preds_array.shape[1] == 1:
        auprc_value = aupr_fn(
            torch.from_numpy(preds_array),
            torch.from_numpy(target_array),
        )
    elif preds_array.shape[1] == 2:
        auprc_value = aupr_fn(
            torch.from_numpy(preds_array[:, 1]),
            torch.from_numpy(target_array),
        )
    return auprc_value


class trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn: torch.nn.modules.loss,
        device: int,
        model_folder_path: str,
        max_epochs: int,
        train_dataloader: torch.utils.data.dataloader,
        valid_dataloader: torch.utils.data.dataloader = None,
        metric: Optional[str] = "pearson",
        counter_for_early_stop_threshold: Optional[int] = 0,
        epochs_to_check_loss: Optional[int] = 0,
        n_accumulated_batches: Optional[int] = 1,
        use_scheduler: Optional[bool] = False,
        **kwargs,
    ):
        self.best_model = model
        self.model = model
        self.model_folder_path = model_folder_path
        self.max_epochs = max_epochs
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.metric = metric
        self.counter_for_early_stop_threshold = counter_for_early_stop_threshold
        self.epochs_to_check_loss = epochs_to_check_loss
        self.n_accumulated_batches = n_accumulated_batches
        self.use_scheduler = use_scheduler
        self.best_valid_loss = np.inf
        self.best_metric_value = -1.0
        self.counter_for_early_stop = 0
        self.logger = configure_logging_format(file_path=self.model_folder_path)
        self.is_distributed = is_dist_avail_and_initialized()
        if (
            self.n_accumulated_batches > len(self.train_dataloader)
            or self.n_accumulated_batches == -1
        ):
            self.n_accumulated_batches = len(train_dataloader)

    def train(self):
        if self.is_distributed:
            dist.barrier()
        model_path, checkpoints_path, loss_csv_path, loss_plot_path = get_paths(
            datetime.now(), self.model_folder_path
        )
        self.logger.info(
            f"Device: {self.device} - Training for {self.max_epochs} epochs"
        )
        self.logger.info(
            f"Device: {self.device} - Number of training batches is {len(self.train_dataloader)} batches"
        )
        self.logger.info(
            f"Device: {self.device} - Number of accumulated batches is {self.n_accumulated_batches} batches"
        )
        self.logger.info(f"Device: {self.device} - Started training")
        early_stopping_flag = torch.zeros(1, device=self.device)
        for epoch in progress_bar(range(1, self.max_epochs + 1)):
            if self.is_distributed:
                dist.all_reduce(early_stopping_flag, op=dist.ReduceOp.SUM)
            if early_stopping_flag.item() == 1:
                break
            train_loss_per_epoch = self.train_loop()
            if self.is_distributed:
                dist.all_reduce(train_loss_per_epoch, op=dist.ReduceOp.SUM)
                train_loss_per_epoch = (
                    train_loss_per_epoch.item() / dist.get_world_size()
                )
            else:
                train_loss_per_epoch = train_loss_per_epoch.item()
            if self.use_scheduler:
                self.optimizer.update_lr(epoch)
            # if is_main_process():
            # Initialize loss
            valid_loss_per_epoch = float("nan")
            metric_value = float("nan")
            if (
                (epoch % self.epochs_to_check_loss == 0)
                and (self.counter_for_early_stop_threshold > 0)
                and self.valid_dataloader != None
            ):
                valid_loss_per_epoch, metric_value = self.get_valid_loss()
                if self.is_distributed:
                    dist.all_reduce(valid_loss_per_epoch, op=dist.ReduceOp.SUM)
                    dist.all_reduce(metric_value, op=dist.ReduceOp.SUM)
                    valid_loss_per_epoch = (
                        valid_loss_per_epoch.item() / dist.get_world_size()
                    )
                    metric_value = metric_value.item() / dist.get_world_size()
                else:
                    valid_loss_per_epoch = valid_loss_per_epoch.item()
                    metric_value = metric_value.item()
            if is_main_process():
                self.counter_for_early_stop += 1
                save_data_to_csv(
                    data_dictionary={
                        "epoch": epoch,
                        "time": datetime.now(),
                        "train_loss": train_loss_per_epoch,
                        "valid_loss": valid_loss_per_epoch,
                        self.metric: metric_value,
                    },
                    csv_file_path=loss_csv_path,
                )
                if metric_value > self.best_metric_value:
                    self.counter_for_early_stop = 0
                    self.best_metric_value = metric_value
                    self.best_model = deepcopy(self.model)
                    torch.save(
                        self.best_model.state_dict(),
                        os.path.join(checkpoints_path, f"model_{epoch}.pt"),
                    )
                elif (
                    self.counter_for_early_stop == self.counter_for_early_stop_threshold
                ):
                    self.logger.info(
                        f"Device: {self.device} - Early stopping at epoch {epoch}"
                    )
                    early_stopping_flag += 1

        if is_main_process():
            self.logger.info(f"Device: {self.device} - Finished training")
            torch.save(self.best_model.state_dict(), model_path)
            plot_loss(loss_csv_path=loss_csv_path, loss_path=loss_plot_path)
        return self.best_model, model_path

    def train_loop(self):
        # Executes a single epoch of training.

        self.model.train()
        loss_per_epoch = torch.tensor(0.0, device=self.device)
        self.optimizer.zero_grad()
        for batch_idx, ((_, data, target), bit) in enumerate(
            progress_bar(self.train_dataloader)
        ):
            data, target = {key: data[key].to(self.device) for key in data}, target.to(
                self.device
            )
            pred = self.model(**data, bit=bit)
            if self.is_distributed:
                loss = self.loss_fn(
                    pred, target  # , list(self.model.module.get_convlayers(bit))
                )
            else:
                loss = self.loss_fn(
                    pred, target
                )  # , list(self.model.get_convlayers(bit)))
            # normalize loss to account for batch accumulation
            loss_per_epoch += loss
            loss = loss / self.n_accumulated_batches
            loss.backward()
            # weights update
            if ((batch_idx + 1) % self.n_accumulated_batches == 0) or (
                batch_idx + 1 == len(self.train_dataloader)
            ):
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
        return loss_per_epoch / (batch_idx + 1)

    def get_valid_loss(self) -> float:
        """
        get the average loss function on valid dataloader
        """
        y_pred = []
        y_true = []
        with torch.no_grad():
            self.model.eval()
            valid_loss = torch.tensor(0.0, device=self.device)
            for batch_idx, ((_, data, target), bit) in enumerate(self.valid_dataloader):
                data, target = {
                    key: data[key].to(self.device) for key in data
                }, target.to(self.device)
                output = self.model(**data, bit=bit)
                loss = self.loss_fn(output, target)
                valid_loss += loss
                if isinstance(output, tuple):
                    output = output[0]
                if self.metric == "pearson" or self.metric == "auprc":
                    y_pred.append(output)
                    y_true.append(target)
        if self.metric == "pearson":
            pred = torch.cat(y_pred, dim=0)
            true = torch.cat(y_true, dim=0)
            pred = pred.detach().cpu().numpy()
            true = true.detach().cpu().numpy()
            metric_value = pearson(pred, true).to(self.device)
        elif self.metric == "auprc":
            pred = torch.cat(y_pred, dim=0)
            true = torch.cat(y_true, dim=0)
            pred = pred.detach().cpu().numpy()
            true = true.detach().cpu().numpy()
            metric_value = auprc(pred, true).to(self.device)
        elif self.metric == "loss":
            metric_value = -valid_loss / (batch_idx + 1)
        else:
            metric_value = None
        return valid_loss / (batch_idx + 1), metric_value
