from typing import Optional, List, Any
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import EsmTokenizer
from src.utils import (
    hot_encode_sequence,
    read_excel_csv_file,
)
from Bio.Seq import reverse_complement
from random import randint
from src.seed import set_seed


set_seed()


def get_masked_sequence(sequence: str, n_regions: int) -> str:
    sequence = np.frombuffer(
        sequence.encode("ascii"), dtype="S1"
    ).copy()  # make writable
    size = sequence.shape[0]
    data = np.random.normal(loc=0, scale=1, size=size)
    indices = np.argpartition(data, -n_regions)[-n_regions:]

    sequence[indices] = b"N"
    return sequence.tobytes().decode("ascii")


def get_augmented_sequences(sequence):
    reverse_complement_seq = reverse_complement(sequence)
    random_int = randint(1, 3)
    sequence_shifted_right = random_int * "N" + sequence[:-random_int]
    sequence_shifted_left = sequence[random_int:] + random_int * "N"
    reverse_sequence_shifted_right = (
        random_int * "N" + reverse_complement_seq[:-random_int]
    )
    reverse_sequence_shifted_left = (
        reverse_complement_seq[random_int:] + random_int * "N"
    )
    masked_sequence = get_masked_sequence(sequence, 25)
    masked_reverse_sequence = get_masked_sequence(reverse_complement_seq, 25)
    return (
        sequence,
        reverse_complement_seq,
        sequence_shifted_right,
        sequence_shifted_left,
        reverse_sequence_shifted_right,
        reverse_sequence_shifted_left,
        masked_sequence,
        masked_reverse_sequence,
    )


class DatasetLoad(Dataset):
    def __init__(
        self,
        sequences_df,
        labels_path,
        length_after_padding: Optional[int] = 0,
        lazyLoad: Optional[bool] = False,
        device: Optional[Any] = "cpu",
        consistency_regularization: Optional[bool] = True,
    ):
        self.lazyLoad = lazyLoad
        self.length_after_padding = length_after_padding
        self.labels_path = labels_path
        self.device = device
        self.consistency_regularization = consistency_regularization
        self.sequences_df = sequences_df.reset_index(drop=True)
        self.records = self.sequences_df["record"].tolist()
        self.sequences = self.sequences_df["sequence"].tolist()
        if not lazyLoad:
            self.data = [
                torch.from_numpy(
                    hot_encode_sequence(
                        sequence=seq,
                        length_after_padding=length_after_padding,
                    )
                )
                for seq in self.sequences
            ]
            self.labels = [
                torch.from_numpy(
                    np.load(os.path.join(self.labels_path, f"{record}.npy"))
                )
                for record in self.records
            ]

        self.len = self.sequences_df.shape[0]

    def __getitem__(self, idx):
        record = self.records[idx]
        if not self.lazyLoad:
            return (
                record,
                dict(input=self.data[idx]),
                self.labels[idx],
            )
        else:
            sequence = self.sequences[idx]
            labels = torch.from_numpy(
                np.load(os.path.join(self.labels_path, f"{record}.npy"))
            )
            if self.consistency_regularization:
                return (
                    [record] * 8,
                    dict(
                        input=torch.from_numpy(
                            np.array(
                                [
                                    hot_encode_sequence(
                                        seq,
                                        length_after_padding=self.length_after_padding,
                                    )
                                    for seq in get_augmented_sequences(sequence)
                                ]
                            ),
                        ),
                    ),
                    labels.repeat(8, 1),
                )
            else:
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


class DatasetLoad_Kmer(Dataset):
    def __init__(
        self,
        sequences_df,
        labels_path,
        length_after_padding: Optional[int] = 0,
        lazyLoad: Optional[bool] = False,
        device: Optional[Any] = "cpu",
        tokenizer: Optional[EsmTokenizer] = None,
        consistency_regularization: Optional[bool] = True,
        add_special_tokens: Optional[bool] = True,
        **kwargs,
    ):

        self.lazyLoad = lazyLoad
        self.length_after_padding = length_after_padding
        self.labels_path = labels_path
        self.device = device
        self.sequences_df = sequences_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.consistency_regularization = consistency_regularization
        self.records = self.sequences_df["record"].tolist()
        self.sequences = self.sequences_df["sequence"].tolist()

        if not lazyLoad:
            if self.consistency_regularization:
                self.data = [
                    self.tokenizer(
                        get_augmented_sequences(seq),
                        return_tensors="pt",
                        add_special_tokens=self.add_special_tokens,
                    )
                    for seq in self.sequences
                ]
                self.labels = [
                    torch.from_numpy(
                        np.load(os.path.join(self.labels_path, f"{record}.npy"))
                    ).repeat(8, 1)
                    for record in self.records
                ]
            else:
                self.data = [
                    self.tokenizer(
                        seq,
                        return_tensors="pt",
                        add_special_tokens=self.add_special_tokens,
                    )
                    for seq in self.sequences
                ]
                self.labels = [
                    torch.from_numpy(
                        np.load(os.path.join(self.labels_path, f"{record}.npy"))
                    )
                    for record in self.records
                ]

        self.len = self.sequences_df.shape[0]

    def __getitem__(self, idx):
        record = self.records[idx]
        if not self.lazyLoad:
            return (
                record,
                self.data[idx],
                self.labels[idx],
            )
        else:
            sequence = self.sequences[idx]
            labels = torch.from_numpy(
                np.load(os.path.join(self.labels_path, f"{record}.npy"))
            )
            if self.consistency_regularization:
                augmented_seqs = get_augmented_sequences(sequence)
                return (
                    [record] * 8,
                    self.tokenizer(
                        augmented_seqs,
                        return_tensors="pt",
                        add_special_tokens=self.add_special_tokens,
                    ),
                    labels.repeat(8, 1),
                )
            else:
                return (
                    record,
                    self.tokenizer(
                        sequence,
                        return_tensors="pt",
                        add_special_tokens=self.add_special_tokens,
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
        sequences_paths: List[str],
        labels_paths: List[str],
    ):
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
        tokenizer: Optional[EsmTokenizer] = None,
        consistency_regularization: Optional[bool] = True,
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
                tokenizer=tokenizer,
                consistency_regularization=consistency_regularization,
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
        tokenizer: Optional[EsmTokenizer] = None,
        consistency_regularization: Optional[bool] = True,
        **kwargs,
    ):
        for indices_path, labels_path in zip(indices_paths, self.labels_paths):
            indices = open(indices_path).read().splitlines()
            if tokenizer == None:
                dataset = DatasetLoad(
                    sequences_df=self.sequences_df.loc[
                        self.sequences_df.record.isin(indices)
                    ],
                    labels_path=labels_path,
                    length_after_padding=length_after_padding,
                    lazyLoad=lazyLoad,
                    device=device,
                    consistency_regularization=consistency_regularization,
                )
            else:
                dataset = DatasetLoad_Kmer(
                    sequences_df=self.sequences_df.loc[
                        self.sequences_df.record.isin(indices)
                    ],
                    labels_path=labels_path,
                    length_after_padding=length_after_padding,
                    lazyLoad=lazyLoad,
                    device=device,
                    tokenizer=tokenizer,
                    consistency_regularization=consistency_regularization,
                    **kwargs,
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
