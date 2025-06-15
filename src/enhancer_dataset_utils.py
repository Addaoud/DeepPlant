from typing import Optional, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from src.utils import (
    hot_encode_sequence,
    read_excel_csv_file,
)
from transformers import EsmTokenizer
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


class EnhancerDatasetLoad(Dataset):
    def __init__(
        self,
        sequences_list,
        labels_list,
        use_reverse_complement: Optional[bool] = False,
        length_after_padding: Optional[int] = 2500,
        lazyLoad: Optional[bool] = False,
    ):
        self.lazyLoad = lazyLoad
        self.length_after_padding = length_after_padding
        self.sequences = sequences_list
        self.labels = labels_list
        if use_reverse_complement:
            for i in range(len(sequences_list)):
                self.sequences.append(reverse_complement(sequences_list[i]))
                self.labels.append(self.labels[i])
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
        self.len = len(self.labels)

    def __getitem__(self, idx):
        if not self.lazyLoad:
            return (
                "_",
                dict(input=self.data[idx].float()),
                self.labels[idx],
            )
        else:
            return (
                "_",
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


class load_dataset:
    def __init__(
        self,
        sequences_paths: List[str],
    ):
        """
        Loads and processes the data.
        """
        self.sequences_df = pd.concat(
            [read_excel_csv_file(sequences_path) for sequences_path in sequences_paths]
        )

    def get_dataloader(
        self,
        dataset: str,
        lazyLoad: Optional[bool] = True,
        shuffle: Optional[bool] = True,
        use_reverse_complement: Optional[bool] = False,
        batchSize: Optional[int] = 8,
        length_after_padding: Optional[int] = 2500,
        num_workers: Optional[int] = 0,
        n_gpu: Optional[int] = 0,
        tokenizer: Optional[EsmTokenizer] = None,
        **kwargs,
    ):
        dataloaders_list = list(
            self.get_dataloaders_list(
                dataset=dataset,
                lazyLoad=lazyLoad,
                shuffle=shuffle,
                use_reverse_complement=use_reverse_complement,
                batchSize=batchSize,
                length_after_padding=length_after_padding,
                num_workers=num_workers,
                n_gpu=n_gpu,
                tokenizer=tokenizer,
                **kwargs,
            )
        )
        return MultiTaskDataLoader(dataloaders_list, shuffler=np.random.RandomState(42))

    def get_dataloaders_list(
        self,
        dataset: str,
        lazyLoad: Optional[bool] = True,
        shuffle: Optional[bool] = True,
        use_reverse_complement: Optional[bool] = False,
        batchSize: Optional[int] = 8,
        length_after_padding: Optional[int] = 0,
        device: Optional[int] = 0,
        num_workers: Optional[int] = 0,
        n_gpu: Optional[int] = 0,
        tokenizer: Optional[EsmTokenizer] = None,
        **kwargs,
    ):
        if tokenizer == None:
            dataset = EnhancerDatasetLoad(
                sequences_list=self.sequences_df.loc[
                    self.sequences_df.dataset == dataset
                ].Sequence.values.tolist(),
                labels_list=self.sequences_df.loc[
                    self.sequences_df.dataset == dataset
                ].Label.values.tolist(),
                use_reverse_complement=use_reverse_complement,
                length_after_padding=length_after_padding,
                lazyLoad=lazyLoad,
            )
        else:
            pass
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
