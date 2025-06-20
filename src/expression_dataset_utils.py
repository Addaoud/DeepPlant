from typing import Optional, List, Any
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from src.utils import (
    hot_encode_sequence,
    read_excel_csv_file,
)
from src.seed import set_seed

set_seed()


class DatasetLoad(Dataset):
    def __init__(
        self,
        sequences_df,
        labels_path,
        length_after_padding: Optional[int] = 0,
        lazyLoad: Optional[bool] = False,
        device: Optional[Any] = "cpu",
    ):
        self.lazyLoad = lazyLoad
        self.length_after_padding = length_after_padding
        self.labels_path = labels_path
        self.device = device
        self.sequences_df = sequences_df.reset_index(drop=True)
        if not lazyLoad:
            self.data = list()
            self.labels = list()
            for idx in range(0, self.sequences_df.shape[0]):
                self.data.append(
                    torch.tensor(
                        hot_encode_sequence(
                            sequence=self.sequences_df.iloc[idx].sequence,
                            length_after_padding=length_after_padding,
                        )
                    )
                )
                self.labels.append(
                    np.load(
                        f"{os.path.join(self.labels_path,self.sequences_df.iloc[idx].header)}.npy"
                    )
                )

        self.len = self.sequences_df.shape[0]

    def __getitem__(self, idx):
        if not self.lazyLoad:
            return (
                self.sequences_df.iloc[idx].header,
                dict(input=self.data[idx].float()),
                self.labels[idx].astype(np.float32),
            )
        else:
            labels = torch.from_numpy(
                np.load(
                    f"{os.path.join(self.labels_path,self.sequences_df.iloc[idx].header.replace('_r',''))}.npy"
                )
            )
            return (
                self.sequences_df.iloc[idx].header,
                dict(
                    input=torch.tensor(
                        hot_encode_sequence(
                            sequence=self.sequences_df.iloc[idx].sequence,
                            length_after_padding=self.length_after_padding,
                        ),
                        dtype=torch.float,
                    ),
                ),
                labels.float(),
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
    def __init__(self, sequences_paths: List[str], labels_paths: List[str]):
        """
        Loads and processes the data.
        """
        self.sequences_df = pd.concat(
            [read_excel_csv_file(sequences_path) for sequences_path in sequences_paths]
        )
        self.labels_paths = labels_paths

    def get_dataloader(
        self,
        indices_paths: List[str],
        lazyLoad: Optional[bool] = True,
        shuffle: Optional[bool] = True,
        batchSize: Optional[int] = 8,
        length_after_padding: Optional[int] = 2048,
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
        for indices_path, label_path in zip(indices_paths, self.labels_paths):
            indices = open(indices_path).read().splitlines()
            dataset = DatasetLoad(
                sequences_df=self.sequences_df.loc[
                    self.sequences_df.header.isin(indices)
                ],
                labels_path=label_path,
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
