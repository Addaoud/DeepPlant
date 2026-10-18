import pandas as pd
import sys

from Bio import motifs
import os
from typing import Optional
import matplotlib.pyplot as plt
import logomaker
import pandas as pd
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.cuda import device_count
from typing import Optional, Any
from src.ddp import setup, cleanup, is_main_process
import torch.distributed as dist
from src.utils import hot_encode_sequence, create_path, save_data_to_csv
import torch
from fastprogress import progress_bar
from src.config import ExpressionConfig
from src.utils import read_json, get_device
from src.DeepPlant_expression import build_model
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from Bio.motifs.matrix import FrequencyPositionMatrix


def load_motif_database(
    db_path: Optional[
        str
    ] = "/s/chromatin/m/nobackup/ahmed/DeepPlant/data/arabidopsis/PFMs",  # "/s/chromatin/a/nobackup/ahmed/AT_motifs/PFMs",
):
    motifs_dict = dict()
    for pfm_file in os.listdir(db_path):
        motifs_dict[pfm_file] = motifs.read(
            open(os.path.join(db_path, pfm_file)), "pfm"
        )
    return motifs_dict


def load_meme_txt_database(meme_path: str):
    """
    Load a MEME *text* format file (MEME v4.x) and return a dictionary of
    Biopython Motif objects keyed by motif name, with PFMs stored in
    motif.counts.
    """
    motifs_dict = {}

    with open(meme_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("MOTIF"):
            parts = line.split()
            motif_id = parts[1]
            motif_name = parts[2] if len(parts) > 2 else motif_id

            # Next line: letter-probability matrix header
            i += 1
            header = lines[i]
            if not header.startswith("letter-probability matrix"):
                raise ValueError(f"Expected matrix header after MOTIF {motif_id}")

            # Extract width and nsites
            w = int(header.split("w=")[1].split()[0])
            nsites = float(header.split("nsites=")[1].split()[0])

            # Read probability matrix (w rows, 4 columns)
            probs = []
            for _ in range(w):
                i += 1
                probs.append([float(x) for x in lines[i].split()])

            probs = np.array(probs).T  # shape (4, w)

            # Convert probabilities -> counts (PFM)
            counts = probs * nsites

            # Build Biopython motif with PFM
            m = motifs.Motif(alphabet="ACGT")
            m.counts = FrequencyPositionMatrix(
                m.alphabet,
                {
                    "A": counts[0],
                    "C": counts[1],
                    "G": counts[2],
                    "T": counts[3],
                },
            )
            m.id = motif_id
            m.name = motif_name

            motifs_dict[motif_name] = m

        i += 1

    return motifs_dict


def get_motifs(motifs_dict, background):
    pssm_dict = dict()
    for tf, motif in motifs_dict.items():
        pwm = motif.counts.normalize(pseudocounts=0.5)
        pssm = pwm.log_odds(background=background)  # this converts pfm to pssm
        pssm_dict[tf] = pssm

    return pssm_dict


def scan_variant_effects_from_dict(
    gene_name,
    position,
    reference,
    mutation,
    ref_seq,
    alt_seq,
    motif_dict,
    score_threshold=0.0,
    csv_path="",
):
    """
    Plots variant effects using raw dictionary PSSM data.

    Args:
        motif_dict: Dict of dicts, e.g. {'MotifName': {'A': [...], 'C': [...]}}
        score_threshold: The log-odds score threshold (e.g., 5.0).
                         Since your values go down to -11, 0 is a reasonable
                         start (better than random), but >5 is usually 'strong binding'.
    """

    # --- 1. Identify Variant ---
    variant_idx = -1
    min_len = min(len(ref_seq), len(alt_seq))
    for i in range(min_len):
        if ref_seq[i] != alt_seq[i]:
            variant_idx = i
            break

    if variant_idx == -1:
        if len(ref_seq) != len(alt_seq):
            variant_idx = min_len  # Handle indel at end
        else:
            # print("No variant found.")
            return

    # print(f"Variant found at index {variant_idx}")

    # --- 2. Helper: Manual PSSM Scoring ---
    def scan_sequence(sequence, matrix_data, thresh):
        """
        Manually scans a sequence with a weight matrix.
        Returns: (best_score, best_start_position)
        """
        # Get motif length from the 'A' list
        motif_len = len(matrix_data["A"])
        seq_len = len(sequence)

        # Safety Check: If motif is longer than sequence, we can't scan
        if motif_len > seq_len:
            return -float("inf"), -1

        # Convert dict to easier lookup format
        # keys: A, C, G, T. values: lists of weights

        best_score = -float("inf")
        best_pos = -1
        for pos, score in matrix_data.search(
            sequence, threshold=matrix_data.max * thresh
        ):
            return (pos, score)

        return best_score, best_pos

    # --- 3. Scan All Motifs ---
    affected_motifs = []

    for name, matrix_data in motif_dict.items():
        pos_ref, score_ref = scan_sequence(ref_seq, matrix_data, score_threshold)
        pos_alt, score_alt = scan_sequence(alt_seq, matrix_data, score_threshold)
        # Logic for Gain/Loss
        # print(score_alt,matrix_data)
        is_hit_ref = score_ref >= (matrix_data.max * score_threshold)
        is_hit_alt = score_alt >= (matrix_data.max * score_threshold)
        if is_hit_ref and not is_hit_alt:
            affected_motifs.append((name, "Loss", matrix_data, pos_ref))
            save_data_to_csv(
                {
                    "gene": gene_name,
                    "position": position,
                    "reference": reference,
                    "mutation": mutation,
                    "reference sequence": ref_seq,
                    "alternative sequence": alt_seq,
                    "motif": name,
                    "motif_pos": pos_ref,
                    "status": "loss",
                },
                csv_path,
            )

        elif not is_hit_ref and is_hit_alt:
            affected_motifs.append((name, "Gain", matrix_data, pos_alt))
            save_data_to_csv(
                {
                    "gene": gene_name,
                    "position": position,
                    "reference": reference,
                    "mutation": mutation,
                    "reference sequence": ref_seq,
                    "alternative sequence": alt_seq,
                    "motif": name,
                    "motif_pos": pos_alt,
                    "status": "gain",
                },
                csv_path,
            )
        # # Optional: Include "Change" if both are hits but score changes drastically
        # elif is_hit_ref and is_hit_alt and abs(score_ref - score_alt) > 3.0:
        #      effect = "Weaker" if score_alt < score_ref else "Stronger"
        #      affected_motifs.append((name, f"{effect} (Change)", matrix_data, pos_alt))

    if not affected_motifs:
        # print("No motifs crossed the threshold criteria at the variant site.")
        return


def main(
    device: Any,
    n_gpu: Optional[int] = 0,
    data_class: Optional[Any] = None,
    ism: Optional[bool] = False,
    scan: Optional[bool] = True,
):
    results_dir = "/s/chromatin/a/nobackup/ahmed/DeepPlant/results/ISM"
    create_path(results_dir)

    # variants_h5="/s/chromatin/a/nobackup/ahmed/DeepPlant/Arabidopsis/phenotype/gene_variants_by_cultivar.h5"
    # h5_file = h5py.File(variants_h5, "r")
    # genome_IDs = h5_file.keys()

    dataloader = data_class.get_dataloader(
        device=device,
        n_gpu=n_gpu,
    )
    if ism:
        config = ExpressionConfig(
            **read_json(
                json_path="/s/chromatin/m/nobackup/ahmed/DeepPlant/json/config_AT_expressionRC.json"
            )
        )
        eps = 1e-4
        model = build_model(
            args=config,
            new_model=False,
            model_path="/s/chromatin/a/nobackup/ahmed/DeepPlant/results/expression_AT/803010/checkpoints/model_25_12_15:05:32.pt",
        ).to(device)
        if n_gpu > 1:
            setup(device, n_gpu)
            model = DDP(
                model,
                device_ids=[device],
                find_unused_parameters=config.find_unused_parameters,
            )
            dist.barrier()
        nuclt = ["A", "C", "G", "T"]
        with torch.no_grad():
            model.eval()
            for _, (gene, sequence, ref_input) in enumerate(progress_bar(dataloader)):
                gene = gene[0]
                ref_allele = sequence[0]
                sum_lfc = torch.zeros(4, 2500).to(device)
                ref_labels = model(input=ref_input)
                for idx in range(len(ref_allele)):
                    for j, nucl in enumerate(nuclt):
                        if ref_allele[idx] != nucl:
                            mut_allele = ref_allele[:idx] + nucl + ref_allele[idx + 1 :]
                            mut_input = (
                                torch.tensor(
                                    hot_encode_sequence(mut_allele), dtype=torch.float
                                )
                                .unsqueeze(0)
                                .to(device)
                            )
                            mut_label = model(input=mut_input)
                            sum_lfc[j, idx] = (
                                torch.log(mut_label + eps) - torch.log(ref_labels + eps)
                            ).mean(dim=0)
                sum_lfc = sum_lfc.cpu().numpy()
                rows, cols = np.where(sum_lfc > 0.9 * sum_lfc.max())
                indices = list(zip(rows, cols))
                for idx in indices:
                    # print(f"{ref_allele[idx[1]]} => {nuclt[idx[0]]} at {idx[1]} (lfc={round(sum_lfc[idx[0],idx[1]].item(),4)})")
                    save_data_to_csv(
                        {
                            "gene": gene,
                            "position": idx[1],
                            "reference": ref_allele[idx[1]],
                            "mutation": nuclt[idx[0]],
                            "score": round(sum_lfc[idx[0], idx[1]].item(), 4),
                        },
                        os.path.join(results_dir, "summary_ISM_gr.csv"),
                    )
                rows, cols = np.where(sum_lfc < 0.9 * sum_lfc.min())
                indices = list(zip(rows, cols))
                for idx in indices:
                    # print(f"{ref_allele[idx[1]]} => {nuclt[idx[0]]} at {idx[1]} (lfc={round(sum_lfc[idx[0],idx[1]].item(),4)})")
                    save_data_to_csv(
                        {
                            "gene": gene,
                            "position": idx[1],
                            "reference": ref_allele[idx[1]],
                            "mutation": nuclt[idx[0]],
                            "score": round(sum_lfc[idx[0], idx[1]].item(), 4),
                        },
                        os.path.join(results_dir, "summary_ISM_ls.csv"),
                    )
    dist.barrier()
    if scan:
        motifs_dict = load_motif_database()
        back_freq = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
        pssm_dict_jaspar = get_motifs(motifs_dict, back_freq)
        motifs_dict = load_motif_database(
            db_path="/s/chromatin/a/nobackup/ahmed/AT_motifs/PFMs"
        )
        pssm_dict_plantTFDB = get_motifs(motifs_dict, back_freq)
        motifs_dict = load_meme_txt_database(
            "/s/chromatin/m/nobackup/ahmed/DeepPlant/haoxuan/ArabidopsisDAPv1.meme"
        )
        pssm_dict_DAPv1 = get_motifs(motifs_dict, back_freq)
        pad = 10
        score_threshold = 0.8
        summary_df_gr = pd.read_csv(os.path.join(results_dir, "summary_ISM_gr.csv"))
        summary_df_ls = pd.read_csv(os.path.join(results_dir, "summary_ISM_ls.csv"))
        for _, (gene, sequence, ref_input) in enumerate(progress_bar(dataloader)):
            gene = gene[0]
            ref_allele = sequence[0]
            for idx, row in summary_df_gr.loc[summary_df_gr.gene == gene].iterrows():
                pos = row.position
                mutation = row.mutation
                ref_DNAsequence = ref_allele[pos - pad : pos + pad]
                mutant_DNAsequence = (
                    ref_allele[pos - pad : pos]
                    + mutation
                    + ref_allele[pos + 1 : pos + pad]
                )
                scan_variant_effects_from_dict(
                    gene,
                    pos,
                    ref_allele[pos],
                    mutation,
                    ref_DNAsequence,
                    mutant_DNAsequence,
                    pssm_dict_jaspar,
                    score_threshold=score_threshold,
                    csv_path=os.path.join(results_dir, "motifs_ISM_jaspar_gr.csv"),
                )
                scan_variant_effects_from_dict(
                    gene,
                    pos,
                    ref_allele[pos],
                    mutation,
                    ref_DNAsequence,
                    mutant_DNAsequence,
                    pssm_dict_plantTFDB,
                    score_threshold=score_threshold,
                    csv_path=os.path.join(results_dir, "motifs_ISM_plantTFDB_gr.csv"),
                )
                scan_variant_effects_from_dict(
                    gene,
                    pos,
                    ref_allele[pos],
                    mutation,
                    ref_DNAsequence,
                    mutant_DNAsequence,
                    pssm_dict_DAPv1,
                    score_threshold=score_threshold,
                    csv_path=os.path.join(results_dir, "motifs_ISM_DAPv1_gr.csv"),
                )
            for idx, row in summary_df_ls.loc[summary_df_ls.gene == gene].iterrows():
                pos = row.position
                mutation = row.mutation
                ref_DNAsequence = ref_allele[pos - pad : pos + pad]
                mutant_DNAsequence = (
                    ref_allele[pos - pad : pos]
                    + mutation
                    + ref_allele[pos + 1 : pos + pad]
                )
                scan_variant_effects_from_dict(
                    gene,
                    pos,
                    ref_allele[pos],
                    mutation,
                    ref_DNAsequence,
                    mutant_DNAsequence,
                    pssm_dict_jaspar,
                    score_threshold=score_threshold,
                    csv_path=os.path.join(results_dir, "motifs_ISM_jaspar_ls.csv"),
                )
                scan_variant_effects_from_dict(
                    gene,
                    pos,
                    ref_allele[pos],
                    mutation,
                    ref_DNAsequence,
                    mutant_DNAsequence,
                    pssm_dict_plantTFDB,
                    score_threshold=score_threshold,
                    csv_path=os.path.join(results_dir, "motifs_ISM_plantTFDB_ls.csv"),
                )
                scan_variant_effects_from_dict(
                    gene,
                    pos,
                    ref_allele[pos],
                    mutation,
                    ref_DNAsequence,
                    mutant_DNAsequence,
                    pssm_dict_DAPv1,
                    score_threshold=score_threshold,
                    csv_path=os.path.join(results_dir, "motifs_ISM_DAPv1_ls.csv"),
                )

    if n_gpu > 1:
        cleanup()


def get_gene(header):
    return header.split("_")[0]


def get_tss(header):
    return int(header.split("_")[2])


class DatasetLoad(Dataset):
    def __init__(self, genes, sequences):
        self.genes = genes
        self.sequences = sequences
        # print(self.genes[0])

    def __len__(self):
        return len(self.genes)

    def __getitem__(self, idx):
        # print(idx,self.genes[idx])
        return (
            self.genes[idx],
            self.sequences[idx],
            torch.from_numpy(
                hot_encode_sequence(
                    sequence=self.sequences[idx], length_after_padding=2500
                )
            ),
        )


class load_dataset:
    def __init__(
        self,
    ):
        expression_df = pd.read_csv(
            "/s/chromatin/m/nobackup/ahmed/DeepPlant/data/arabidopsis/expression/expression_data_TSS_centered_2500_0.csv"
        )

        expression_df["gene"] = list(map(get_gene, expression_df.header))
        expression_df["tss"] = list(map(get_tss, expression_df.header))
        df = pd.read_csv(
            "/s/chromatin/m/nobackup/ahmed/DeepPlant/data/arabidopsis/arabidopsis_expression_data_split.csv"
        )
        self.test_genes = df.loc[df.split == "test"].gene.values
        self.gene_df = expression_df.loc[expression_df.gene.isin(self.test_genes)]

    def get_dataloader(
        self,
        device: Optional[int] = 0,
        n_gpu: Optional[int] = 0,
    ):
        num_workers = 8
        dataset = DatasetLoad(
            self.gene_df.gene.values.tolist(), self.gene_df.sequence.values.tolist()
        )
        sampler = DistributedSampler(dataset, num_replicas=n_gpu, rank=device)
        if n_gpu > 1:
            sampler = DistributedSampler(dataset, num_replicas=n_gpu, rank=device)
            return DataLoader(
                dataset,
                batch_size=1,
                sampler=sampler,
                pin_memory=True,
                num_workers=num_workers,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                pin_memory=True,
                num_workers=num_workers,
            )


if __name__ == "__main__":
    device = get_device()
    data_class = load_dataset()

    if device == "cuda":
        n_gpu = device_count()
        # logger.info(f"Using {n_gpu} gpu(s)")
        mp.spawn(
            main,
            args=(
                n_gpu,
                data_class,
            ),
            nprocs=n_gpu,
            join=True,
        )
    else:
        main(
            device=device,
            n_gpu=0,
            data_class=data_class,
        )

    main()
