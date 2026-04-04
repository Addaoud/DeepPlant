# %%
import torch
import collections
from DeepPlant_full_new import build_model, DeepPlantConfig
import argparse, os
import numpy as np
import pandas as pd
import logging
from typing import Literal, List
from datetime import datetime
from scipy.stats import pearsonr
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import json
import argparse
import os

# pretrain
# Argument parsing
# pretrain


def parse_args():
    p = argparse.ArgumentParser(
        description="DeepPlant motif extraction using pretraining dataset"
    )
    p.add_argument(
        "--data_masked_csv",
        type=str,
        default="/s/chromatin/c/nobackup/deepplant/Data/Arabidopsis_thaliana/Non_Overlap_avg_2500_200_50_old/2500_Seq_10kb_masked.csv",
        help="Masked pretraining sequence CSV",
    )

    p.add_argument(
        "--train_id_txt",
        type=str,
        default="/s/chromatin/c/nobackup/deepplant/Data/Arabidopsis_thaliana/Non_Overlap_avg_2500_200_50_old/Test.txt",
        help="Text file listing seq IDs to use",
    )

    p.add_argument(
        "--model_dir",
        type=str,
        default="/s/chromatin/m/nobackup/ahmed/DeepPlant/haoxuan",
        help="Directory to save CNN filter motif results",
    )

    p.add_argument(
        "--config_json",
        type=str,
        default="/s/chromatin/m/nobackup/ahmed/DeepPlant/haoxuan/config_AT_2500.json",
        help="DeepPlant model config JSON",
    )

    p.add_argument("--seq_len", type=int, default=2500)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--threshold_ratio", type=float, default=0.9)

    return p.parse_args()


# pretrain
# Utility Functions
# pretrain


def fatonumpy(fa_sequence: str) -> np.ndarray:
    """
    Convert DNA sequence in string to one-hot encoded sequence
    Returns shape: (4, L)
    """
    nucleotide_to_index = {"A": 0, "C": 1, "G": 2, "T": 3}
    L = len(fa_sequence)
    one_hot_seq = np.zeros((L, 4))
    idx = np.array([nucleotide_to_index.get(n, -1) for n in fa_sequence])
    mask = idx >= 0
    one_hot_seq[np.arange(L)[mask], idx[mask]] = 1
    return one_hot_seq.T


# pretrain
# Motif helper functions
# pretrain
import matplotlib.pyplot as plt
import seaborn as sns
import logomaker


def save_selected_seqs_to_fasta(selected_seqs, output_path="motifs.fasta"):
    with open(output_path, "w") as f:
        for i, (seq, (score, idx)) in enumerate(selected_seqs):
            header = f">seq_{i}|score={score:.4f}|index={idx}"
            f.write(header + "\n")
            f.write(seq + "\n")
    print(f"Saved {len(selected_seqs)} sequences to {output_path}")


def plot_pwm_from_selected_seqs(selected_seqs, save_prefix="motif"):
    base_order = ["A", "C", "G", "T"]
    kernel_size = len(selected_seqs[0][0])  # motif length

    counts = np.zeros((kernel_size, 4))
    for motif_str, (score, _) in selected_seqs:
        for i, base in enumerate(motif_str):
            if base in base_order:
                counts[i, base_order.index(base)] += score  # weighted by score

    pwm = counts / counts.sum(axis=1, keepdims=True)
    df_pwm = pd.DataFrame(pwm, columns=base_order)

    # PWM logo
    plt.figure(figsize=(kernel_size // 2, 2))
    logomaker.Logo(df_pwm)
    plt.title("Motif PWM (weighted)")
    plt.xlabel("Position")
    plt.ylabel("Frequency")
    plt.savefig(f"{save_prefix}_pwm.png", dpi=300, bbox_inches="tight")
    plt.close()

    # IC logo
    s = 4
    n = len(selected_seqs)
    en = (1 / np.log(2)) * (s - 1) / (2 * n)

    with np.errstate(divide="ignore", invalid="ignore"):
        Hi = -np.nansum(pwm * np.log2(pwm), axis=1)
        Hi = np.nan_to_num(Hi)

    Ri = np.log2(s) - Hi - en
    logo_mat = pwm * Ri[:, None]
    df_logo = pd.DataFrame(logo_mat, columns=base_order)

    plt.figure(figsize=(kernel_size // 2, 2))
    logomaker.Logo(df_logo)
    plt.title("Motif Information Content")
    plt.xlabel("Position")
    plt.ylabel("Bits")
    plt.savefig(f"{save_prefix}_ic.png", dpi=300, bbox_inches="tight")
    plt.close()


# pretrain
# Main Script
# pretrain
args = parse_args()
os.makedirs(args.model_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# pretrain
# Load model config and model
# pretrain
with open(args.config_json) as f:
    cfg_json = json.load(f)
config = DeepPlantConfig(**cfg_json)

model = build_model(
    False,
    model_path="/s/chromatin/a/nobackup/ahmed/DeepPlant/results/DeepPlant_AT/177318/model_25_12_18:05:28.pt",
    json_path=args.config_json,
).to(device)

model.eval()
print("Loaded pretrained model DeepPlantAT.pt")

first_conv = model.backbone.conv_features[0][0]  # Conv1d

# pretrain
# Load pretraining masked sequence dataset
# pretrain
print("\nLoading Train.txt ...")
with open(args.train_id_txt, "r") as f:
    train_ids = set(line.strip() for line in f)
print(f"Train.txt contains {len(train_ids)} IDs.")

print("Loading masked CSV ...")
df = pd.read_csv(args.data_masked_csv)

# Normalize column name
if df.columns[0] != "seq_id":
    df.rename(columns={df.columns[0]: "seq_id"}, inplace=True)

# Detect sequence column
if "sequence" in df.columns:
    seq_col = "sequence"
else:
    seq_col = df.columns[1]

df = df[df["seq_id"].isin(train_ids)]
print(f"Using {len(df)} sequences for motif extraction")

# pretrain
# Convert to one-hot
# pretrain
seqLen = args.seq_len
onehot_seqs = []

for _, row in df.iterrows():
    seq = str(row[seq_col])
    if len(seq) < seqLen:
        seq = seq + "N" * (seqLen - len(seq))
    elif len(seq) > seqLen:
        seq = seq[:seqLen]

    onehot = fatonumpy(seq)
    onehot_seqs.append(onehot)

onehot_seqs = np.stack(onehot_seqs)
print(f"Final onehot shape: {onehot_seqs.shape}")

# pretrain
# CNN filter motif extraction
# pretrain
weights = first_conv.conv.weight.data.cpu().numpy()
num_filters = weights.shape[0]
print(f"\nProcessing {num_filters} CNN filters...\n")

output_base = args.model_dir
motif_dir = os.path.join(output_base, "motifresult")
os.makedirs(motif_dir, exist_ok=True)

batch_size = args.batch_size
num_seqs = onehot_seqs.shape[0]
seq_len = onehot_seqs.shape[2]

for filter_idx in range(num_filters):
    # if filter_idx < 512:
    #     continue
    print(f"=== Filter {filter_idx}/{num_filters} ===")

    filter_dir = os.path.join(motif_dir, f"filter{filter_idx}")
    os.makedirs(filter_dir, exist_ok=True)

    max_acts, max_pos, raw_idx = [], [], []

    with torch.no_grad():
        for i in range(0, num_seqs, batch_size):
            batch = onehot_seqs[i : i + batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)

            act = first_conv(batch_tensor)[:, filter_idx, :]
            batch_max, batch_arg = act.max(dim=1)

            max_acts.extend(batch_max.cpu().numpy())
            max_pos.extend(batch_arg.cpu().numpy())
            raw_idx.extend(range(i, min(i + batch_size, num_seqs)))

    max_acts = np.array(max_acts)
    max_pos = np.array(max_pos)
    raw_idx = np.array(raw_idx)

    # threshold = 80% of global max
    global_max = np.max(max_acts)
    thr = args.threshold_ratio * global_max
    idxs = np.where(max_acts >= thr)[0]

    print(f"  Global max: {global_max:.4f}, threshold: {thr:.4f}")
    print(f"  Sequences above threshold: {len(idxs)}")

    if len(idxs) == 0:
        continue

    window = first_conv.conv.kernel_size[0]
    dedup = {}
    base_map = ["A", "C", "G", "T"]

    for i in idxs:
        seq_i = raw_idx[i]
        pos = max_pos[i]
        score = max_acts[i]

        if pos + window > seq_len:
            continue

        frag = onehot_seqs[seq_i, :, pos : pos + window]
        seq_str = "".join([base_map[np.argmax(frag[:, j])] for j in range(window)])

        if (seq_str not in dedup) or (score > dedup[seq_str][0]):
            dedup[seq_str] = (score, seq_i)

    selected = sorted(dedup.items(), key=lambda x: x[1][0], reverse=True)
    print(f"  Unique sequences kept: {len(selected)}")

    # Save conv filter weights
    motif = weights[filter_idx]
    plt.figure(figsize=(10, 2))
    sns.heatmap(motif, cmap="coolwarm", yticklabels=["A", "C", "G", "T"])
    plt.title(f"Filter {filter_idx} weights")
    plt.savefig(
        os.path.join(filter_dir, f"filter{filter_idx}_weights.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # PWM + IC logo
    if len(selected) != 0:
        prefix = os.path.join(filter_dir, f"filter{filter_idx}")
        plot_pwm_from_selected_seqs(selected, save_prefix=prefix)

        # FASTA
        fasta_path = os.path.join(filter_dir, f"top_motifs_{filter_idx}.fasta")
        save_selected_seqs_to_fasta(selected, fasta_path)

print("\n=== All filters processed successfully! ===")
