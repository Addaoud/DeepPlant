import torch
import numpy as np
import os
from datetime import datetime
from .utils import save_data_to_csv, plot_loss, get_paths
from .seed import set_seed
from .ddp import is_main_process, is_dist_avail_and_initialized
from .logger import configure_logging_format
from fastprogress import progress_bar
from typing import Optional
import torch.distributed as dist
from copy import deepcopy


set_seed()


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
        self.counter_for_early_stop_threshold = counter_for_early_stop_threshold
        self.epochs_to_check_loss = epochs_to_check_loss
        self.n_accumulated_batches = n_accumulated_batches
        self.use_scheduler = use_scheduler
        self.best_valid_loss = np.inf
        self.counter_for_early_stop = 0
        self.logger = configure_logging_format(file_path=self.model_folder_path)
        if (
            self.n_accumulated_batches > len(self.train_dataloader)
            or self.n_accumulated_batches == -1
        ):
            self.n_accumulated_batches = len(train_dataloader)

    def train(self):
        if is_dist_avail_and_initialized():
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
            if is_dist_avail_and_initialized():
                dist.all_reduce(early_stopping_flag, op=dist.ReduceOp.SUM)
            if early_stopping_flag == 1:
                break
            train_loss_per_epoch = self.train_loop()
            if self.use_scheduler:
                self.optimizer.update_lr(epoch)
            # if is_main_process():
            # Initialize loss
            valid_loss_per_epoch = float("nan")
            if (epoch % self.epochs_to_check_loss == 0) and (
                self.counter_for_early_stop_threshold > 0
            ):
                valid_loss_per_epoch = self.get_valid_loss()
            if is_dist_avail_and_initialized():
                train_loss_per_epoch = torch.tensor([train_loss_per_epoch]).to(
                    self.device
                )
                valid_loss_per_epoch = torch.tensor([valid_loss_per_epoch]).to(
                    self.device
                )
                dist.all_reduce(train_loss_per_epoch, op=dist.ReduceOp.SUM)
                dist.all_reduce(valid_loss_per_epoch, op=dist.ReduceOp.SUM)
                train_loss_per_epoch = (
                    train_loss_per_epoch.item() / dist.get_world_size()
                )
                valid_loss_per_epoch = (
                    valid_loss_per_epoch.item() / dist.get_world_size()
                )
            if is_main_process():
                self.counter_for_early_stop += 1
                save_data_to_csv(
                    data_dictionary={
                        "epoch": epoch,
                        "train_loss": train_loss_per_epoch,
                        "valid_loss": valid_loss_per_epoch,
                    },
                    csv_file_path=loss_csv_path,
                )
                if valid_loss_per_epoch < self.best_valid_loss:
                    self.counter_for_early_stop = 0
                    self.best_valid_loss = valid_loss_per_epoch
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
        loss_per_epoch = 0
        self.optimizer.zero_grad()
        for batch_idx, ((_, data, target), bit) in enumerate(
            progress_bar(self.train_dataloader)
        ):
            data, target = {key: data[key].to(self.device) for key in data}, target.to(
                self.device
            )

            pred = self.model(**data, bit=bit)
            if is_dist_avail_and_initialized():
                loss = self.loss_fn(
                    pred, target  # , list(self.model.module.get_convlayers(bit))
                )
            else:
                loss = self.loss_fn(
                    pred, target
                )  # , list(self.model.get_convlayers(bit)))
            # normalize loss to account for batch accumulation
            loss = loss / self.n_accumulated_batches
            loss.backward()
            loss_per_epoch += loss.item() * self.n_accumulated_batches
            # weights update
            if ((batch_idx + 1) % self.n_accumulated_batches == 0) or (
                batch_idx + 1 == len(self.train_dataloader)
            ):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
        return loss_per_epoch / (batch_idx + 1)

    def get_valid_loss(self) -> float:
        """
        get the average loss function on valid dataloader
        """
        with torch.no_grad():
            self.model.eval()
            valid_loss = 0
            for batch_idx, ((_, data, target), bit) in enumerate(self.valid_dataloader):
                data, target = {
                    key: data[key].to(self.device) for key in data
                }, target.to(self.device)
                output = self.model(**data, bit=bit)
                if is_dist_avail_and_initialized():
                    loss = self.loss_fn(output, target)
                else:
                    loss = self.loss_fn(output, target)
                valid_loss += loss.item()
        valid_loss /= batch_idx + 1
        return valid_loss
