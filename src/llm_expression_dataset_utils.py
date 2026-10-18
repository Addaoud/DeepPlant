from typing import Optional, List, Any
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import h5py
from transformers import AutoTokenizer

from src.seed import set_seed

set_seed()


# ----------------------------------------------------------------------------
# Model registry: maps a short model key -> (hf_id, tokenizer_kwargs).
# NT: 6-mer tokenization (~6 bp/token), needs trust_remote_code for the ESM
#     custom code. PlantCAD2: single-nucleotide tokenization, trust_remote_code
#     for the Caduceus/Mamba2 custom code. Both expose a standard HF tokenizer.
# ----------------------------------------------------------------------------
MODEL_REGISTRY = {
    "nt": "InstaDeepAI/NTv3_650M_pre",
    "plantcad2": "kuleshov-group/PlantCAD2-Small-l24-d0768",
    "agront": "InstaDeepAI/agro-nucleotide-transformer-1b",
    "pdllm": "zhangtaolab/plant-dnagpt-6mer",
}

TOKENIZE_KWARGS = {
    "nt": dict(add_special_tokens=False, pad_to_multiple_of=128),  # U-Net: len % 128
    "plantcad2": dict(),  # default special tokens
    "agront": dict(),  # keep <CLS>, as in pre-training
    "pdllm": dict(),  # default special tokens
}


def resolve_model_id(model_key_or_id: str) -> str:
    """Accept either a registry key ('nt', 'plantcad2') or a raw HF id."""
    return MODEL_REGISTRY.get(model_key_or_id, model_key_or_id)


class LLMDatasetLoad(Dataset):
    """
    Mirrors DatasetLoad's H5 contract (records / sequences / labels datasets,
    optional indices_path filtering, lazy per-worker H5 init) but tokenizes raw
    sequence strings with a HuggingFace tokenizer instead of one-hot encoding.

    Returns per item:
        (record, dict(input_ids=LongTensor, attention_mask=LongTensor), labels)

    Tokenization to a fixed length is done here (padding/truncation) so the
    default collate can stack directly and the trainer needs no custom collate.
    """

    def __init__(
        self,
        h5_path,
        tokenizer,
        max_length: int,
        tokenize_kwargs: dict,
        indices_path: Optional[Any] = None,
        device: Optional[Any] = "cpu",
    ):
        self.h5_path = h5_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.indices_path = indices_path
        self.device = device
        self.h5 = None
        self.records = None
        self.sequences = None
        self.labels = None
        self.tokenize_kwargs = tokenize_kwargs

        # Open file briefly to get ALL record names and identify indices
        with h5py.File(self.h5_path, "r") as f:
            all_records = [r.decode("ascii") for r in f["records"][:]]

            if self.indices_path and os.path.exists(self.indices_path):
                with open(self.indices_path, "r") as t:
                    wanted_records = set(line.strip() for line in t if line.strip())
                self.valid_indices = [
                    i for i, name in enumerate(all_records) if name in wanted_records
                ]
            else:
                self.valid_indices = list(range(len(all_records)))

        self.len = len(self.valid_indices)

    def _init_h5(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_path, "r", swmr=True, libver="latest")
            self.records = self.h5["records"]
            self.sequences = self.h5["sequences"]
            self.labels = self.h5["labels"]

    def _tokenize(self, sequence: str):
        enc = self.tokenizer(
            sequence.upper(),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            **self.tokenize_kwargs,
        )
        input_ids = enc["input_ids"].squeeze(0)
        if "attention_mask" in enc:
            attention_mask = enc["attention_mask"].squeeze(0)
        else:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return input_ids, attention_mask

    def __getitem__(self, idx):
        self._init_h5()
        real_idx = self.valid_indices[idx]
        record = self.records[real_idx].decode("ascii")
        sequence = self.sequences[real_idx].decode("ascii")
        labels = torch.from_numpy(np.log1p(self.labels[real_idx])).float()

        input_ids, attention_mask = self._tokenize(sequence)
        return (
            record,
            dict(input_ids=input_ids, attention_mask=attention_mask),
            labels,
        )

    def __len__(self):
        return self.len


class MultiTaskDataLoader(DataLoader):
    """Data loader-like object that combines and samples from multiple single-task data loaders."""

    def __init__(self, dataloaders: List[DataLoader], shuffler=None):
        self.dataloaders = dataloaders
        self.shuffler = shuffler
        self.d_min = min([len(dl) for dl in self.dataloaders])

    def __len__(self):
        return self.d_min * len(self.dataloaders)

    def set_epoch(self, epoch):
        for dl in self.dataloaders:
            if isinstance(dl.sampler, DistributedSampler):
                dl.sampler.set_epoch(epoch)

    def __iter__(self):
        task_choice_list = []
        for i, dl in enumerate(self.dataloaders):
            task_choice_list += [i] * self.d_min
        task_choice_list = np.array(task_choice_list)
        if self.shuffler is not None:
            self.shuffler.shuffle(task_choice_list)
        dataloader_iters = [iter(dl) for dl in self.dataloaders]
        for i in task_choice_list:
            yield next(dataloader_iters[i]), i


class load_llm_dataset:
    """
    Drop-in analogue of load_dataset for the LLM benchmark. Construct once with
    the H5 paths and the model key, then call get_dataloader() the same way the
    DeepPlant main script does.
    """

    def __init__(
        self,
        h5_paths: List[str],
        model_name: str,
    ):
        self.h5_paths = h5_paths
        self.model_key = model_name
        self.model_id = resolve_model_id(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.tokenize_kwargs = TOKENIZE_KWARGS.get(self.model_key, {})
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def get_dataloader(
        self,
        indices_paths: List[str],
        shuffle: Optional[bool] = True,
        batchSize: Optional[int] = 8,
        num_workers: Optional[int] = 0,
        n_gpu: Optional[int] = 0,
        max_length: Optional[int] = None,
        device: Optional[Any] = 0,
        **kwargs,
    ):
        # If max_length isn't given explicitly, fall back to the tokenizer's
        # model_max_length (capped to avoid absurd padding for short seqs).
        if max_length is None:
            max_length = min(self.tokenizer.model_max_length, 100_000)

        dataloaders_list = list(
            self.get_dataloaders_list(
                indices_paths=indices_paths,
                shuffle=shuffle,
                batchSize=batchSize,
                num_workers=num_workers,
                n_gpu=n_gpu,
                max_length=max_length,
                device=device,
            )
        )
        return MultiTaskDataLoader(dataloaders_list, shuffler=np.random.RandomState(42))

    def get_dataloaders_list(
        self,
        indices_paths: List[str],
        shuffle: Optional[bool] = True,
        batchSize: Optional[int] = 8,
        num_workers: Optional[int] = 0,
        n_gpu: Optional[int] = 0,
        max_length: Optional[int] = 1000,
        device: Optional[int] = 0,
        **kwargs,
    ):
        for indices_path, h5_path in zip(indices_paths, self.h5_paths):
            dataset = LLMDatasetLoad(
                h5_path=h5_path,
                tokenizer=self.tokenizer,
                max_length=max_length,
                tokenize_kwargs=self.tokenize_kwargs,
                indices_path=indices_path,
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
