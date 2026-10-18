# %%
from src.DeepPlant import build_model
from src.config import CSPConfig
import argparse, os
import numpy as np
import pandas as pd
import json
import argparse
import os
import torch

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
        default="/s/chromatin/c/nobackup/deepplant/Data/Arabidopsis_thaliana/Non_Overlap_avg_2500_200_50_old/Train.txt",
        help="Text file listing seq IDs to use",
    )

    p.add_argument(
        "--model_dir",
        type=str,
        default="/s/chromatin/m/nobackup/ahmed/GitRepositries/DeepPlant/analysis/CNN_filters",
        help="Directory to save CNN filter motif results",
    )

    p.add_argument(
        "--config_json",
        type=str,
        default="/s/chromatin/m/nobackup/ahmed/GitRepositries/DeepPlant/config/config_AT_CSP.json",
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
    kernel_size = len(selected_seqs[0][0])

    counts = np.zeros((kernel_size, 4))
    for motif_str, _ in selected_seqs:
        for i, base in enumerate(motif_str):
            if base in base_order:
                counts[i, base_order.index(base)] += 1  # unweighted: +1 per occurrence

    # guard against empty columns before normalizing
    col_sums = counts.sum(axis=1, keepdims=True)
    pwm = np.divide(
        counts, col_sums, out=np.full_like(counts, 0.25), where=col_sums > 0
    )
    df_pwm = pd.DataFrame(pwm, columns=base_order)

    plt.figure(figsize=(kernel_size // 2, 2))
    logomaker.Logo(df_pwm)
    plt.title("Motif PWM")
    plt.xlabel("Position")
    plt.ylabel("Frequency")
    plt.savefig(f"{save_prefix}_pwm.png", dpi=300, bbox_inches="tight")
    plt.close()

    s = 4
    n = sum(counts[0])  # actual number of aligned occurrences
    en = (1 / np.log(2)) * (s - 1) / (2 * n) if n > 0 else 0
    with np.errstate(divide="ignore", invalid="ignore"):
        Hi = -np.nansum(pwm * np.log2(pwm), axis=1)
        Hi = np.nan_to_num(Hi)
    Ri = np.log2(s) - Hi - en
    df_logo = pd.DataFrame(pwm * Ri[:, None], columns=base_order)

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
config = CSPConfig(**cfg_json)

model = build_model(
    args=config,
    new_model=False,
    model_path="/s/chromatin/m/nobackup/ahmed/GitRepositries/DeepPlant/models/model_AT_CSP.pt",
    task="CSP",
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

df = df[df["seq_id"].isin(train_ids)][:100000]
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
window = first_conv.conv.kernel_size[0]
base_map = ["A", "C", "G", "T"]

for filter_idx in range(num_filters):
    filter_dir = os.path.join(motif_dir, f"filter{filter_idx}")
    if not os.path.exists(filter_dir):
        print(f"=== Filter {filter_idx}/{num_filters} ===")

        os.makedirs(filter_dir, exist_ok=True)

        # ---- pass 1: gather the full activation distribution (all positions) ----
        all_acts = []
        with torch.no_grad():
            for i in range(0, num_seqs, batch_size):
                batch = torch.tensor(
                    onehot_seqs[i : i + batch_size], dtype=torch.float32
                ).to(device)
                act = first_conv(batch)[:, filter_idx, :]  # (B, L_out)
                all_acts.append(act.cpu().numpy().ravel())
        all_acts = np.concatenate(all_acts)

        # distribution-based threshold, consistent in meaning across filters:
        #   half-max between this filter's min and max activation (Basset convention)
        a_min, a_max = all_acts.min(), all_acts.max()
        if a_max <= 0:  # filter never fires positively -> skip
            print("  filter never activates positively; skipped")
            continue
        thr = a_min + args.threshold_ratio * (a_max - a_min)
        print(f"  min={a_min:.4f} max={a_max:.4f} thr={thr:.4f}")

        # ---- pass 2: extract EVERY window above threshold (all positions) ----
        dedup = {}
        with torch.no_grad():
            for i in range(0, num_seqs, batch_size):
                batch_np = onehot_seqs[i : i + batch_size]
                batch = torch.tensor(batch_np, dtype=torch.float32).to(device)
                act = first_conv(batch)[:, filter_idx, :].cpu().numpy()  # (B, L_out)

                hit_rows, hit_pos = np.where(act >= thr)  # all hits, not just argmax
                for b, pos in zip(hit_rows, hit_pos):
                    if pos + window > seq_len:
                        continue
                    frag = batch_np[b, :, pos : pos + window]
                    # skip windows overlapping N-padding (all-zero columns)
                    if np.any(frag.sum(axis=0) == 0):
                        continue
                    seq_str = "".join(
                        base_map[np.argmax(frag[:, j])] for j in range(window)
                    )
                    score = act[b, pos]
                    if (seq_str not in dedup) or (score > dedup[seq_str][0]):
                        dedup[seq_str] = (score, i + b)

        selected = sorted(dedup.items(), key=lambda x: x[1][0], reverse=True)
        print(f"  windows above threshold (unique): {len(selected)}")
        if not selected:
            continue

        # filter weight heatmap
        plt.figure(figsize=(10, 2))
        sns.heatmap(
            weights[filter_idx], cmap="coolwarm", yticklabels=["A", "C", "G", "T"]
        )
        plt.title(f"Filter {filter_idx} weights")
        plt.savefig(
            os.path.join(filter_dir, f"filter{filter_idx}_weights.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        prefix = os.path.join(filter_dir, f"filter{filter_idx}")
        plot_pwm_from_selected_seqs(selected, save_prefix=prefix)
        save_selected_seqs_to_fasta(
            selected, os.path.join(filter_dir, f"top_motifs_{filter_idx}.fasta")
        )

print("\n=== All filters processed successfully! ===")
