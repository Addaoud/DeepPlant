from typing import Optional, List, Any
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from src.utils import (
    hot_encode_sequence,
)
import h5py

from src.seed import set_seed

set_seed()


class DatasetLoad(Dataset):
    def __init__(
        self,
        h5_path,
        indices_path: Optional[str] = None,
        length_after_padding: Optional[int] = 0,
        lazyLoad: Optional[bool] = False,
        device: Optional[Any] = "cpu",
        consistency_regularization: Optional[bool] = True,
    ):
        self.h5_path = h5_path
        self.indices_path = indices_path
        self.lazyLoad = lazyLoad
        self.length_after_padding = length_after_padding
        self.device = device
        self.consistency_regularization = consistency_regularization
        self.h5 = None
        self.records = None
        self.sequences = None
        self.labels = None
        self.n_augment = 6
        # Open file briefly to get ALL record names and identify indices
        with h5py.File(self.h5_path, "r") as f:
            all_records = [r.decode("ascii") for r in f["records"][:]]

            if self.indices_path and os.path.exists(self.indices_path):
                # If a Train.txt is provided, filter the indices
                with open(self.indices_path, "r") as t:
                    wanted_records = set(line.strip() for line in t if line.strip())

                # Map the record name to its position in the H5 file
                self.valid_indices = [
                    i for i, name in enumerate(all_records) if name in wanted_records
                ]
            else:
                # If no file provided, use everything
                self.valid_indices = list(range(len(all_records)))

        self.len = len(self.valid_indices)

    def _init_h5(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_path, "r", swmr=True, libver="latest")
            self.records = self.h5["records"]
            self.sequences = self.h5["sequences"]
            self.labels = self.h5["labels"]

    def __getitem__(self, idx):
        self._init_h5()
        real_idx = self.valid_indices[idx]
        record = self.records[real_idx].decode("ascii")
        sequence = self.sequences[real_idx].decode("ascii")
        labels = torch.from_numpy(np.log1p(self.labels[real_idx])).float()
        return (
            record,
            dict(
                input=torch.from_numpy(
                    hot_encode_sequence(
                        sequence,
                        length_after_padding=self.length_after_padding,
                    ),
                ),
            ),
            labels,
        )

    def __len__(self):
        return self.len


class MultiTaskDataLoader(DataLoader):
    """Data loader-like object that combines and samples from multiple single-task data loaders."""

    def __init__(self, dataloaders: List[DataLoader], shuffler=None):
        self.dataloaders = dataloaders
        self.shuffler = shuffler  # reproducible training
        self.d_min = min([len(dl) for dl in self.dataloaders])

    def __len__(self):
        # return sum(len(dl) for dl in self.dataloaders)
        return self.d_min * len(self.dataloaders)

    def __iter__(self):
        """For each batch, sample a task, and yield a batch from the respective task Dataloader."""
        task_choice_list = []
        for i, dl in enumerate(self.dataloaders):
            # task_choice_list += [i] * len(dl)
            task_choice_list += [i] * self.d_min
        task_choice_list = np.array(task_choice_list)
        if self.shuffler is not None:
            self.shuffler.shuffle(task_choice_list)
        dataloader_iters = [iter(dl) for dl in self.dataloaders]
        for i in task_choice_list:
            yield next(dataloader_iters[i]), i


class load_dataset:
    def __init__(
        self,
        h5_paths: List[str],
    ):
        """
        Loads and processes the data.
        """
        self.h5_paths = [
            os.path.join(os.getenv("DEEPPLANTPATH"), h5_path) for h5_path in h5_paths
        ]

    def get_dataloader(
        self,
        indices_paths: List[str],
        lazyLoad: Optional[bool] = True,
        shuffle: Optional[bool] = True,
        batchSize: Optional[int] = 8,
        length_after_padding: Optional[int] = 2500,
        num_workers: Optional[int] = 0,
        n_gpu: Optional[int] = 0,
        **kwargs,
    ):
        dataloaders_list = list(
            self.get_dataloaders_list(
                indices_paths=indices_paths,
                lazyLoad=lazyLoad,
                shuffle=shuffle,
                batchSize=batchSize,
                length_after_padding=length_after_padding,
                num_workers=num_workers,
                n_gpu=n_gpu,
                **kwargs,
            )
        )
        return MultiTaskDataLoader(dataloaders_list, shuffler=np.random.RandomState(42))

    def get_dataloaders_list(
        self,
        indices_paths: List[str],
        lazyLoad: Optional[bool] = True,
        shuffle: Optional[bool] = True,
        batchSize: Optional[int] = 8,
        length_after_padding: Optional[int] = 0,
        device: Optional[int] = 0,
        num_workers: Optional[int] = 0,
        n_gpu: Optional[int] = 0,
        **kwargs,
    ):
        for indices_path, h5_path in zip(indices_paths, self.h5_paths):
            dataset = DatasetLoad(
                h5_path=h5_path,
                indices_path=indices_path,
                length_after_padding=length_after_padding,
                lazyLoad=lazyLoad,
                device=device,
            )
            if n_gpu > 1:
                sampler = DistributedSampler(dataset, num_replicas=n_gpu, rank=device)
                yield DataLoader(
                    dataset,
                    batch_size=batchSize,
                    sampler=sampler,
                    pin_memory=True,
                    num_workers=num_workers,
                )
            else:
                yield DataLoader(
                    dataset,
                    batch_size=batchSize,
                    shuffle=shuffle,
                    pin_memory=True,
                    num_workers=num_workers,
                )
