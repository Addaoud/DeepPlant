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
from datetime import datetime
from scipy.stats import pearsonr
from contextlib import nullcontext

set_seed()


def _gather_cat(x: torch.Tensor) -> torch.Tensor:
    """Concatenate a tensor across all ranks (CPU). No-op when not distributed.
    Must be called by every rank."""
    if not is_dist_avail_and_initialized():
        return x.cpu()
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, x.detach().cpu())
    return torch.cat(out)


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


def pearson(preds_array: np.ndarray, target_array: np.ndarray) -> torch.Tensor:
    """Mean per-target Pearson correlation."""
    if target_array.ndim == 3:
        target_array = target_array.reshape(-1, target_array.shape[2])
    pearsonr_list = [
        pearsonr(preds_array[:, i], target_array[:, i])[0]
        for i in range(target_array.shape[1])
    ]
    return torch.tensor([np.nanmean(pearsonr_list)], dtype=torch.float32)


def auprc(preds_array: np.ndarray, target_array: np.ndarray) -> torch.Tensor:
    """AUPRC on class-1 probabilities (logits are converted first)."""
    scores = _to_scores(torch.as_tensor(preds_array).float())
    target = torch.as_tensor(target_array).long().reshape(-1)
    return AveragePrecision(task="binary")(scores, target).reshape(1)


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
        self.model = model
        self.best_state = None
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
        self.best_metric_value = -np.inf
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
            if hasattr(self.train_dataloader, "set_epoch"):
                self.train_dataloader.set_epoch(epoch)
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
                valid_loss_per_epoch = valid_loss_per_epoch.to(self.device)
                metric_value = metric_value.to(self.device)
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
                    self.best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                    torch.save(
                        self.best_state,
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
            if self.best_state is not None:
                self.model.load_state_dict(self.best_state)
            else:
                self.logger.info(
                    f"Device: {self.device} - No validation improvement recorded; "
                    "keeping the final model"
                )
            torch.save(self.model.state_dict(), model_path)
            plot_loss(loss_csv_path=loss_csv_path, loss_path=loss_plot_path)
        return self.model, model_path

    def train_loop(self):
        # Executes a single epoch of training.

        self.model.train()
        loss_per_epoch = torch.tensor(0.0, device=self.device)
        self.optimizer.zero_grad()
        for batch_idx, ((_, data, target), bit) in enumerate(
            progress_bar(self.train_dataloader)
        ):
            data, target = {
                key: data[key].to(self.device, non_blocking=True) for key in data
            }, target.to(self.device, non_blocking=True)
            is_accumulating = ((batch_idx + 1) % self.n_accumulated_batches != 0) and (
                batch_idx + 1 != len(self.train_dataloader)
            )
            sync_context = (
                self.model.no_sync()
                if (self.is_distributed and is_accumulating)
                else nullcontext()
            )
            with sync_context:
                pred = self.model(**data, bit=bit)
                loss = self.loss_fn(pred, target)
                loss_per_epoch += loss.detach()
                loss = loss / self.n_accumulated_batches
                loss.backward()
            # weights update
            if not is_accumulating:
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
        return loss_per_epoch / (batch_idx + 1)

    def get_valid_loss(self) -> float:
        """
        get the average loss function on valid dataloader
        """
        y_pred, y_true = [], []
        valid_loss = torch.tensor(0.0, device=self.device)
        with torch.no_grad():
            self.model.eval()
            for batch_idx, ((_, data, target), bit) in enumerate(self.valid_dataloader):
                data = {key: data[key].to(self.device) for key in data}
                target = target.to(self.device)
                output = self.model(**data, bit=bit)
                loss = self.loss_fn(output, target)
                valid_loss += loss
                if isinstance(output, tuple):
                    output = output[0]
                if self.metric in ("pearson", "auprc"):
                    y_pred.append(output.detach().cpu())
                    y_true.append(target.detach().cpu())
        valid_loss = valid_loss / (batch_idx + 1)

        if self.metric == "pearson":
            pred = _gather_cat(torch.cat(y_pred, dim=0)).numpy()
            true = _gather_cat(torch.cat(y_true, dim=0)).numpy()
            metric_value = pearson(pred, true)
        elif self.metric == "auprc":
            pred = _gather_cat(torch.cat(y_pred, dim=0)).numpy()
            true = _gather_cat(torch.cat(y_true, dim=0)).numpy()
            metric_value = auprc(pred, true)
        elif self.metric == "loss":
            metric_value = -valid_loss.detach().cpu().reshape(1)
        else:
            raise ValueError(f"Unknown validation metric: {self.metric}")
        return valid_loss, metric_value
