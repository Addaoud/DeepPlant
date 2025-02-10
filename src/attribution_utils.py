from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
from operator import and_
from Bio.Seq import reverse_complement
from .utils import hot_encode_sequence, read_excel_csv_file
from fastprogress import progress_bar
from captum.attr import IntegratedGradients
from typing import Optional
from torch.utils.data import DataLoader
import os

sns.set_theme()


class IGDataset(Dataset):
    def __init__(
        self,
        df_final: pd.DataFrame,
        sampler: Optional[np.array] = None,
        relevant_target: Optional[int] = None,
    ):
        self.df_final = df_final
        if len(sampler) != 0:
            self.df_final = self.df_final.iloc[sampler, :].reset_index(drop=True)
        for (
            idx,
            row,
        ) in self.df_final.iterrows():
            self.df_final.loc[len(self.df_final.index)] = [
                row.header + "_r",
                reverse_complement(row.sequence),
                row.label,
            ]
        if relevant_target:
            self.df_final = self.df_final.loc[
                self.df_final.label.isin([relevant_target])
            ]
        self.df_final = self.df_final.reset_index()
        self.One_hot_Encoded_Tensors = []
        self.Label_Tensors = torch.tensor(self.df_final["label"].tolist())
        for i in range(0, self.df_final.shape[0]):
            self.One_hot_Encoded_Tensors.append(
                torch.tensor(hot_encode_sequence(sequence=self.df_final["sequence"][i]))
            )

    def __len__(self):
        return self.df_final.shape[0]

    def __getitem__(self, idx):
        return (
            self.df_final.header[idx],
            self.df_final.sequence[idx],
            self.One_hot_Encoded_Tensors[idx],
            self.Label_Tensors[idx].long(),
        )


def get_IGdata():
    test_sampler = np.loadtxt("data/test_indices.txt", dtype=int)
    valid_sampler = np.loadtxt("data/valid_indices.txt", dtype=int)
    sampler = np.concatenate((test_sampler, valid_sampler))
    df_final = pd.read_csv("data/final_data.csv")
    IG_dataset_0 = IGDataset(
        df_final=df_final,
        sampler=sampler,
        relevant_target=0,
    )
    IG_loader_0 = DataLoader(IG_dataset_0, batch_size=16)
    IG_dataset_1 = IGDataset(
        df_final=df_final,
        sampler=sampler,
        relevant_target=1,
    )
    IG_loader_1 = DataLoader(IG_dataset_1, batch_size=16)
    DNAalphabet = {"A": 0, "C": 0, "G": 0, "T": 0}
    for _, row in df_final.iterrows():
        for nt in DNAalphabet.keys():
            DNAalphabet[nt] = DNAalphabet[nt] + row.sequence.count(nt)
    sum_nt = sum(DNAalphabet.values())
    for nt in DNAalphabet.keys():
        DNAalphabet[nt] = DNAalphabet[nt] / sum_nt
    back_freq = np.array(list(DNAalphabet.values())).reshape(4, 1)
    return IG_loader_0, IG_loader_1, back_freq, DNAalphabet


def motif_indices(IG_matrix, IG_window_size: int, IG_threshhold: float):
    l = IG_matrix.shape[-1]
    thresh = IG_matrix.max() * IG_threshhold
    last_pos = 0
    pad = 1  # int(IG_window_size / 3)
    for i in range(l - int(IG_window_size / 2)):
        if ((i - last_pos) > pad) and (
            IG_matrix[:, i + int(IG_window_size / 2)].max() > thresh
        ):
            last_pos = i
            yield (i, IG_matrix[:, i + int(IG_window_size / 2)].max())


def plot_motif_heat(param_matrix, file_path: str):
    param_range = abs(param_matrix).max()
    sns.set_theme(font_scale=2)
    plt.figure(figsize=(param_matrix.shape[1], 4))
    sns.heatmap(
        param_matrix, cmap="PRGn", linewidths=0.2, vmin=-param_range, vmax=param_range
    )
    ax = plt.gca()
    ax.set_xticklabels(range(1, param_matrix.shape[1] + 1))
    ax.set_yticklabels("ACGT", rotation="horizontal")
    plt.savefig(file_path)
    plt.close()


def extract_seq(model, dataLoader, windowSize, igThreshhold, relevantTarget, device):
    sequences = []
    IGs = []
    Scores = []
    headers = []
    positions = []
    n_unique_sequences = 0
    baseline = 0.25 * torch.ones((1, 4, 600)).to(device)
    with torch.no_grad():
        model.eval()
        integrated_gradients = IntegratedGradients(model)
        for _, (header, seqs, data, target) in enumerate(progress_bar(dataLoader)):
            data = data.to(device, dtype=torch.float)
            outputs = model(data)
            softmax = torch.nn.Softmax(dim=1)
            pred = softmax(outputs).detach().cpu()
            prediction_score, pred_label_idx = torch.topk(pred, k=1)
            thresh_indices = torch.gt(prediction_score, 0.8).squeeze()
            label_indices = pred_label_idx.squeeze() == target
            indices = and_(thresh_indices, label_indices)
            if sum(indices) != 0:
                n_unique_sequences += sum(indices)
                attributions_ig = integrated_gradients.attribute(
                    data[indices],
                    target=relevantTarget,
                    baselines=baseline,
                    n_steps=20,
                )
                attributions_ig = attributions_ig.cpu().detach().numpy()
                seqs = np.array(seqs)[indices]
                header = np.array(header)[indices]
                for n in range(attributions_ig.shape[0]):
                    for bps in list(
                        motif_indices(
                            attributions_ig[n, :, :],
                            IG_window_size=windowSize,
                            IG_threshhold=igThreshhold,
                        )
                    ):
                        start_pos = bps[0]
                        end_pos = start_pos + windowSize
                        score = bps[1]
                        sequences.append(seqs[n][start_pos:end_pos])
                        if attributions_ig[n, :, start_pos:end_pos].shape == (
                            4,
                            windowSize,
                        ):
                            IGs.append(attributions_ig[n, :, start_pos:end_pos])
                        else:
                            temp_arr = np.zeros((4, windowSize))
                            st = (
                                windowSize
                                - attributions_ig[n, :, start_pos:end_pos].shape[1]
                            )
                            temp_arr[
                                :,
                                st : st
                                + attributions_ig[n, :, start_pos:end_pos].shape[1],
                            ]
                            IGs.append(temp_arr)
                        Scores.append(score)
                        positions.append(int(start_pos))
                        headers.append(header[n])
    return (headers, positions, sequences, IGs, Scores)


def mat_product(motif, hotencoded_DNAsequence, threshold=0.8):
    if (
        scipy.signal.convolve2d(hotencoded_DNAsequence, motif, mode="valid") > threshold
    ).any():
        return True
    return False


def get_motif(mat):
    bp_dict = {0: "A", 1: "C", 2: "G", 3: "T"}
    motif = ""
    for i in range(mat.shape[1]):
        motif += bp_dict[np.argmax(mat[:, i])]
    return motif
