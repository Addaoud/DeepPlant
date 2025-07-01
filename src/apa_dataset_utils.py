from typing import Optional, List, Any
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import EsmTokenizer
from src.utils import (
    hot_encode_sequence,
    read_fasta_file,
)
from Bio.Seq import reverse_complement
from src.seed import set_seed


set_seed()


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


class apa_dataset(Dataset):
    def __init__(
        self,
        records_list,
        sequences_list,
        labels_list,
        length_after_padding: Optional[int] = 2500,
        lazyLoad: Optional[bool] = False,
    ):
        self.lazyLoad = lazyLoad
        self.length_after_padding = length_after_padding
        self.records = records_list
        self.sequences = sequences_list
        self.labels = labels_list
        if not lazyLoad:
            self.data = []
            for idx in range(0, len(self.sequences)):
                self.data.append(
                    torch.tensor(
                        hot_encode_sequence(
                            sequence=self.sequences[idx],
                            length_after_padding=length_after_padding,
                        )
                    )
                )
        self.len = len(self.records)

    def __getitem__(self, idx):
        if not self.lazyLoad:
            return (
                self.records[idx],
                dict(input=self.data[idx].float()),
                self.labels[idx],
            )
        else:
            return (
                self.records[idx],
                dict(
                    input=torch.tensor(
                        hot_encode_sequence(
                            sequence=self.sequences[idx],
                            length_after_padding=self.length_after_padding,
                        )
                    ).float()
                ),
                self.labels[idx],
            )

    def __len__(self):
        return self.len


def get_apa_dataset(
    fasta_path: str,
    lazyLoad: Optional[bool] = True,
    shuffle: Optional[bool] = True,
    batchSize: Optional[int] = 8,
    length_after_padding: Optional[int] = 2500,
    device: Optional[int] = 0,
    num_workers: Optional[int] = 0,
    n_gpu: Optional[int] = 0,
):
    record_list = list()
    sequence_list = list()
    labels_list = list()
    for record in read_fasta_file(fasta_path):
        record_list.append(record.description)
        sequence_list.append(record.seq)
        labels_list.append(int(record.description.split("|")[1]))
    dataset = apa_dataset(
        records_list=record_list,
        sequences_list=sequence_list,
        labels_list=labels_list,
        length_after_padding=length_after_padding,
        lazyLoad=lazyLoad,
    )
    if n_gpu > 1:
        sampler = DistributedSampler(dataset, num_replicas=n_gpu, rank=device)
        dataloader = DataLoader(
            dataset,
            batch_size=batchSize,
            sampler=sampler,
            pin_memory=True,
            num_workers=num_workers,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batchSize,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
        )
    return MultiTaskDataLoader([dataloader], shuffler=np.random.RandomState(42))
