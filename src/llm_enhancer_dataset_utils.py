from typing import Optional, List, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer
from Bio.Seq import reverse_complement

from src.utils import read_excel_csv_file
from src.seed import set_seed

set_seed()


MODEL_REGISTRY = {
    "nt": "InstaDeepAI/NTv3_650M_pre",
    "plantcad2": "kuleshov-group/PlantCAD2-Small-l24-d0768",
    "agront": "InstaDeepAI/agro-nucleotide-transformer-1b",
    "pdllm": "zhangtaolab/plant-dnagpt-6mer",
}


def resolve_model_id(model_key_or_id: str) -> str:
    return MODEL_REGISTRY.get(model_key_or_id, model_key_or_id)


def is_plantcad(model_key: str, model_id: str) -> bool:
    key = (model_key or "").lower()
    mid = (model_id or "").lower()
    return "plantcad" in key or "plantcad" in mid or "caduceus" in mid


class LLMDatasetLoad(Dataset):
    """
    Enhancer (EAP) dataset for the LLM benchmark. Mirrors EnhancerDatasetLoad's
    CSV contract (seq_idx / Sequence / Label columns, split by `dataset` value)
    but tokenizes raw sequence strings with a HuggingFace tokenizer instead of
    one-hot encoding.

    Tokenization branches on the model:
      NTv3      -> char-level, no special tokens, length padded to a multiple
                   of 128 (U-Net conv-tower requirement).
      PlantCAD2 -> keep default special token(s), no 128-multiple padding
                   (Mamba/Caduceus has no length constraint).

    Returns per item:
        (seq_idx, dict(input_ids=LongTensor, attention_mask=LongTensor), label)
    """

    def __init__(
        self,
        sequences_idx,
        sequences_list,
        labels_list,
        tokenizer,
        max_length: int,
        is_plantcad_model: bool,
        use_reverse_complement: Optional[bool] = False,
    ):
        self.sequences_idx = list(sequences_idx)
        self.sequences = list(sequences_list)
        self.labels = list(labels_list)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_plantcad = is_plantcad_model

        if use_reverse_complement:
            n = len(self.sequences)
            for i in range(n):
                self.sequences.append(reverse_complement(self.sequences[i]))
                self.labels.append(self.labels[i])
                self.sequences_idx.append(self.sequences_idx[i])

        self.len = len(self.labels)

    def _tokenize(self, sequence: str):
        if self.is_plantcad:
            enc = self.tokenizer(
                sequence,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
        else:
            enc = self.tokenizer(
                sequence,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                add_special_tokens=False,
                pad_to_multiple_of=128,
                max_length=self.max_length,
            )
        input_ids = enc["input_ids"].squeeze(0)
        if "attention_mask" in enc:
            attention_mask = enc["attention_mask"].squeeze(0)
        else:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return input_ids, attention_mask

    def __getitem__(self, idx):
        input_ids, attention_mask = self._tokenize(self.sequences[idx])
        label = torch.tensor(self.labels[idx])
        return (
            self.sequences_idx[idx],
            dict(input_ids=input_ids, attention_mask=attention_mask),
            label,
        )

    def __len__(self):
        return self.len


class MultiTaskDataLoader(DataLoader):
    """Combines and samples from multiple single-task data loaders."""

    def __init__(self, dataloaders: List[DataLoader], shuffler=None):
        self.dataloaders = dataloaders
        self.shuffler = shuffler
        self.d_min = min([len(dl) for dl in self.dataloaders])

    def __len__(self):
        return self.d_min * len(self.dataloaders)

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
    Drop-in analogue of the enhancer load_dataset for the LLM benchmark.
    Construct with the CSV paths and the model key, then call get_dataloader()
    with dataset="Train" | "Val" | "Test". Tokenization branches on the model
    (NTv3 vs PlantCAD2) internally.
    """

    def __init__(
        self,
        sequences_paths: List[str],
        model_name: str,
    ):
        self.sequences_df = pd.concat([read_excel_csv_file(p) for p in sequences_paths])
        self.model_key = model_name
        self.model_id = resolve_model_id(model_name)
        self.is_plantcad = is_plantcad(self.model_key, self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )

    def get_dataloader(
        self,
        dataset: str,
        shuffle: Optional[bool] = True,
        use_reverse_complement: Optional[bool] = False,
        batchSize: Optional[int] = 8,
        num_workers: Optional[int] = 0,
        n_gpu: Optional[int] = 0,
        max_length: Optional[int] = None,
        device: Optional[Any] = 0,
        **kwargs,
    ):
        if max_length is None:
            max_length = min(self.tokenizer.model_max_length, 100_000)
        dataloaders_list = list(
            self.get_dataloaders_list(
                dataset=dataset,
                shuffle=shuffle,
                use_reverse_complement=use_reverse_complement,
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
        dataset: str,
        shuffle: Optional[bool] = True,
        use_reverse_complement: Optional[bool] = False,
        batchSize: Optional[int] = 8,
        num_workers: Optional[int] = 0,
        n_gpu: Optional[int] = 0,
        max_length: Optional[int] = 1000,
        device: Optional[int] = 0,
        **kwargs,
    ):
        split = self.sequences_df.loc[self.sequences_df.dataset == dataset]
        ds = LLMDatasetLoad(
            sequences_idx=split.seq_idx.values.tolist(),
            sequences_list=split.Sequence.values.tolist(),
            labels_list=split.Label.values.tolist(),
            tokenizer=self.tokenizer,
            max_length=max_length,
            is_plantcad_model=self.is_plantcad,
            use_reverse_complement=use_reverse_complement,
        )
        if n_gpu > 1:
            sampler = DistributedSampler(ds, num_replicas=n_gpu, rank=device)
            yield DataLoader(
                ds,
                batch_size=batchSize,
                sampler=sampler,
                pin_memory=True,
                num_workers=num_workers,
            )
        else:
            yield DataLoader(
                ds,
                batch_size=batchSize,
                shuffle=shuffle,
                pin_memory=True,
                num_workers=num_workers,
            )
