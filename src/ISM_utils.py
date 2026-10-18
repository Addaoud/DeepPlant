import os
from Bio import motifs
import requests
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
import matplotlib.font_manager as fm
from Bio.motifs.matrix import FrequencyPositionMatrix
from fastprogress import progress_bar
import requests
from src.utils import create_path
from functools import lru_cache
from pdf2image import convert_from_path
import logomaker
from math import ceil
import gffutils
from pathlib import Path

# Module-level cache, keyed by DB path so multiple DBs can coexist.
_db_cache: dict[str, gffutils.FeatureDB] = {}


def ppm_to_pwm(ppm, background):
    bg = np.array([background[n] for n in ["A", "C", "G", "T"]])
    pwm = np.log2(ppm / bg)
    return pwm


def load_meme_database(meme_path: str, format: Optional[str] = "pfm"):
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

            if format == "ppm":
                motifs_dict[motif_name] = probs

            elif format == "pfm":
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


def nuclt_dict(sequences):
    back_freq = {"A": 0, "C": 0, "G": 0, "T": 0}
    for sequence in sequences:
        for nuclt in back_freq.keys():
            back_freq[nuclt] += sequence.count(nuclt)
    back_freq = {key: back_freq[key] / sum(back_freq.values()) for key in back_freq}
    return back_freq


def get_motifs(motifs_dict, background):
    pssm_dict = dict()
    for tf, motif in motifs_dict.items():
        pwm = motif.counts.normalize(pseudocounts=0.5)
        pssm = pwm.log_odds(background=background)  # this converts pfm to pssm
        pssm_dict[tf] = pssm

    return pssm_dict


def get_motif_name(matrix_id):
    # matrix_id = "MA1192.2"
    try:
        url = f"https://jaspar.elixir.no/api/v1/matrix/{matrix_id}"

        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        return data["name"]
    except:
        return matrix_id


def get_latest_motifs(motif_dict):
    latest_dict = {}

    for motif in motif_dict.keys():
        # Split 'MA1186.1.TF' -> ['MA1186', '1', 'TF']
        parts = motif.split(".")
        base_id = parts[0]
        version = int(parts[1])

        # If we haven't seen this base ID, or if this version is newer
        if base_id not in latest_dict or version > latest_dict[base_id]["version"]:
            latest_dict[base_id] = {"version": version, "full_name": motif}

    # Extract just the motif names
    return {
        key["full_name"]: motif_dict[key["full_name"]] for key in latest_dict.values()
    }


def relative_to_genomic_position(rel_pos, tss, strand, center=1250):
    """
    Convert relative position (0-2499) in 2500bp TSS-centered sequence
    to genomic coordinate.

    rel_pos: index in sequence (0-based)
    tss: integer TSS coordinate
    strand: '+' or '-'
    """
    rel_pos = int(rel_pos)
    tss = int(tss)
    strand = int(strand)
    genomic_pos = tss + (rel_pos - center) * strand
    return genomic_pos


def extract_chromosome(header):
    for token in header.split():
        if "Chr" in token:
            return token.strip()
    return None


def get_tf_info_ensembl(query_term):
    try:
        species = "arabidopsis_thaliana"
        headers = {"Content-Type": "application/json"}

        # Step 1: resolve symbol → Ensembl gene ID
        url = f"https://rest.ensembl.org/xrefs/symbol/{species}/{query_term}?"
        res = requests.get(url, headers=headers, timeout=20)

        if res.status_code != 200:
            return [query_term]

        data = res.json()
        if not data:
            return [query_term]

        gene_id = data[0].get("id", None)
        if gene_id is None:
            return [query_term]

        # Step 2: fetch gene info
        url2 = f"https://rest.ensembl.org/lookup/id/{gene_id}?"
        res2 = requests.get(url2, headers=headers, timeout=20)

        if res2.status_code != 200:
            return [query_term]

        gene_info = res2.json()

        symbol = gene_info.get("display_name", "Unknown")
        locus = gene_info.get("id", "Unknown")  # Ensembl gene ID

        aliases = [symbol, locus, query_term]

        # Remove duplicates while preserving order
        seen = set()
        aliases = [x for x in aliases if not (x in seen or seen.add(x))]

        return aliases

    except:
        return [query_term]


def get_tf_info_uniprot(query_term):
    try:
        query = f"{query_term} AND organism_id:3702"
        url = f"https://rest.uniprot.org/uniprotkb/search?query={query}&format=json&size=1"

        res = requests.get(url, timeout=20)
        if res.status_code != 200:
            return [query_term]

        data = res.json()
        results = data.get("results", [])
        if not results:
            return [query_term]
        entry = results[0]
        # Extract gene names
        gene_names = entry.get("genes", [])
        aliases = []

        for gene in gene_names:
            if "geneName" in gene:
                aliases.append(gene["geneName"]["value"])
            if "synonyms" in gene:
                for syn in gene["synonyms"]:
                    aliases.append(syn["value"])

        # Extract primary accession (as locus-like identifier)
        locus = entry.get("primaryAccession", "Unknown")

        aliases.append(locus)
        aliases.append(query_term)

        # Remove duplicates
        seen = set()
        aliases = [x for x in aliases if not (x in seen or seen.add(x))]

        return aliases

    except:
        return [query_term]


def get_tf_info(query_term):
    # Arabidopsis taxonomy ID is 3702
    url = f"https://mygene.info/v3/query?q={query_term}&species=3702&fields=symbol,name,alias,locus_tag"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return [query_term]
        data = response.json()
        if "hits" in data and len(data["hits"]) > 0:
            hit = data["hits"][0]
            locus = hit.get("locus_tag", "Unknown")
            symbol = hit.get("symbol", "Unknown")
            aliases = hit.get("alias", [])
            if isinstance(aliases, str):
                aliases = [aliases]
            aliases.append(symbol)
            aliases.append(locus)
            aliases.append(query_term)
            return list(set(aliases))
        return [query_term]
    except:
        return [query_term]


@lru_cache(maxsize=None)
def _cached_read_bed(filepath):
    """Caches loaded BED files in memory to prevent repetitive, slow disk I/O."""
    return pd.read_csv(filepath, delimiter="\t", header=None)


def load_bed_for_tf(tf_name, metadata, bed_files, bed_files_path):
    """
    Returns:
        bed_df (DataFrame) or None if not found
    """
    aliases = get_tf_info(tf_name)
    tf_rows = metadata[metadata.Factor.isin(aliases)]

    if tf_rows.empty:
        return None

    # Define priority of conditions
    conditions = [
        (tf_rows.Mutant == "WT") & (tf_rows.Treatment == "No treatment"),
        (tf_rows.Mutant == "WT"),
        (tf_rows.Treatment == "No treatment"),
        (tf_rows.Type_Strain == "Col-0"),
        pd.Series(True, index=tf_rows.index),  # Fallback to any matching TF row
    ]

    for cond in conditions:
        subset = tf_rows[cond]
        if not subset.empty:
            row = subset.iloc[0]
            accession = row["Sample accession number"]
            experiment_name = row["Experiment Name"]

            # Check for files and use cached reader
            if f"{accession}.target.idr.bed" in bed_files:
                return pd.read_csv(
                    f"{bed_files_path}/{accession}.target.idr.bed",
                    delimiter="\t",
                    header=None,
                )
            elif f"{experiment_name}.target.idr.bed" in bed_files:
                return pd.read_csv(
                    f"{bed_files_path}/{experiment_name}.target.idr.bed",
                    delimiter="\t",
                    header=None,
                )

    return None


def check_peak_overlap(bed_df, chromosome, genomic_pos, window=0):

    overlaps = bed_df[
        (bed_df[0] == chromosome)
        & (bed_df[1] <= genomic_pos + window)
        & (bed_df[2] >= genomic_pos - window)
    ]

    return len(overlaps) > 0


def scan_variant_effects_from_dict(
    ref_seq,
    alt_seq,
    motif_dict,
    score_threshold=0.0,
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
            return -1, -float("inf")

        # Convert dict to easier lookup format
        # keys: A, C, G, T. values: lists of weights

        best_score = -float("inf")
        best_pos = -1
        for pos, score in matrix_data.search(
            sequence, threshold=matrix_data.max * thresh
        ):
            return (pos, score)

        return best_pos, best_score

    # --- 3. Scan All Motifs ---
    affected_motifs = []

    for name, matrix_data in motif_dict.items():
        pos_ref, score_ref = scan_sequence(ref_seq, matrix_data, score_threshold)
        pos_alt, score_alt = scan_sequence(alt_seq, matrix_data, score_threshold)
        # Logic for Gain/Loss
        # print(score_alt,matrix_data)
        is_hit_ref = score_ref >= (matrix_data.max * score_threshold)
        is_hit_alt = score_alt >= (matrix_data.max * score_threshold)
        if is_hit_ref:
            is_hit_alt = score_alt >= (matrix_data.max * 0.5)
            if not is_hit_alt:
                affected_motifs.append(
                    (
                        f"{name}",
                        "Loss",
                        pos_ref,
                    )
                )

        elif is_hit_alt:
            is_hit_ref = score_ref >= (matrix_data.max * 0.5)
            if not is_hit_ref:
                affected_motifs.append(
                    (
                        f"{name}",
                        "Gain",
                        pos_alt,
                    )
                )
    return affected_motifs


def analyze_mutations_with_chip(
    gene,
    sum_lfc,
    ref_sequence,
    nuclt,
    gene_df_full,
    motif_dict,
    metadata,
    bed_files_path,
    mutation_threshold=0.7,
    score_threshold=0.7,
    pad=20,
    seq_len=2500,
):
    nuc_to_index = {"A": 0, "C": 1, "G": 2, "T": 3}
    rows = np.array([nuc_to_index[n] for n in ref_sequence])
    cols = np.arange(len(ref_sequence))
    center = seq_len // 2

    ISM_values = (sum_lfc - sum_lfc.mean(0))[rows, cols]
    results = []

    # Gene information
    row = gene_df_full.loc[gene_df_full.gene == gene].iloc[0]
    strand = str(row.strand)
    tss = int(row.tss)
    chromosome = f"Chr{row.chrom}"

    # Identify strong mutations
    indices = np.where(ISM_values > mutation_threshold * ISM_values.max())[0]

    # Cache the bed directory contents once as a frozenset for fast lookups
    bed_files = frozenset(os.listdir(bed_files_path))

    for pos in progress_bar(indices):
        mutation = nuclt[np.argmin(sum_lfc[:, pos])]

        if ref_sequence[pos] == mutation:
            continue

        # Local sequence window
        ref_local = ref_sequence[max(0, pos - pad) : pos + pad]
        alt_local = (
            ref_sequence[max(0, pos - pad) : pos]
            + mutation
            + ref_sequence[pos + 1 : pos + pad]
        )

        # Scan motif effects
        affected = scan_variant_effects_from_dict(
            ref_local, alt_local, motif_dict, score_threshold
        )

        if not affected:
            continue

        # Convert to genomic coordinate
        genomic_pos = relative_to_genomic_position(
            rel_pos=pos, tss=tss, strand=strand, center=center
        )

        # Check each motif
        for motif_name, effect, motif_rel_pos in affected:
            tf_name = motif_name.split(".")[2]
            motif_strand = "1" if motif_rel_pos > 0 else "-1"
            gene_region = "promoter" if pos < center else "downstream"
            bed_df = load_bed_for_tf(tf_name, metadata, bed_files, bed_files_path)

            if bed_df is None:
                peak_status = "No bed file available"
            else:
                # Optimized fast check
                peak_status = check_peak_overlap(bed_df, chromosome, genomic_pos)

            results.append(
                {
                    "gene": gene,
                    "gene_strand": strand,
                    "chromosome": chromosome,
                    "tss_relative_position": pos - center,
                    "gene_region": gene_region,
                    "sequence_relative_position": pos,
                    "genomic_position": genomic_pos,
                    "ref": ref_sequence[pos],
                    "alt": mutation,
                    "motif": motif_name,
                    "motif_strand": motif_strand,
                    "effect": effect,
                    "chip_peak_overlap": peak_status,
                }
            )

    return pd.DataFrame(results)


def get_size(mutation_pos, pos, region_length):
    for p in mutation_pos:
        if abs(pos - p) < 5:
            return 1.0
    if region_length < 500:
        return 0.5
    elif region_length < 1250:
        return 0.25
    else:
        return 0.1


def draw_letter(ax, letter, x, height, color, size):
    """
    Draw a nucleotide letter at position x.
    Positive height → upright.
    Negative height → flipped vertically (head-down).
    """

    if height == 0:
        return

    text = TextPath(
        (0, 0),
        letter,
        size=size,
        prop=fm.FontProperties(family="DejaVu Sans", weight="bold"),
    )

    if height > 0:
        transform = Affine2D().scale(1, height).translate(x, 0)
    else:
        transform = Affine2D().scale(1, height).translate(x, 0)

    patch = PathPatch(text, lw=0, fc=color, transform=transform + ax.transData)

    ax.add_patch(patch)


def reverse_complement_pwm(ppm):
    # Reverse rows (positions)
    ppm_rc = ppm[::-1, :]
    # Swap columns A,C,G,T -> T,G,C,A
    ppm_rc = ppm_rc[:, [3, 2, 1, 0]]
    return ppm_rc


def plot_pwm(ax, meme_path, motifs_list):
    """
    ax : mosaic axis (panel E)
    meme_file : path to MEME file
    memes : dict {TF_name : motif_id}
    """

    # 1. Clean the main axis (use it only as a container)
    ax.axis("off")

    # 2. Dynamic Grid Calculation
    # We want 2 columns for compact fit. Calculate rows needed.
    n_motifs = len(motifs_list)
    n_cols = 6
    n_rows = int(np.ceil(n_motifs / n_cols))
    w_ratios = [1] * n_cols
    h_ratios = [1] * n_rows

    # 3. Create Subgrid within the Panel
    # hspace=0.6 gives enough room for titles without overlap
    gs = ax.get_subplotspec().subgridspec(
        n_rows,  # Add 2 for top/bottom margins
        n_cols,  # Add 2 for left/right margins
        width_ratios=w_ratios,
        height_ratios=h_ratios,
        wspace=0.5,
        hspace=0.5,
    )

    motifs = load_meme_database(meme_path, format="ppm")

    # 4. Iterate and Plot
    for i, (tf, strand) in enumerate(motifs_list):
        # Calculate grid position
        row = i // n_cols
        col = i % n_cols

        # Add subplot to the specific grid slot
        ax_logo = ax.figure.add_subplot(gs[row, col])

        # Prepare Data
        ppm = motifs[tf].transpose()
        # ppm = motif["matrix"]
        if strand == "-1":
            ppm = reverse_complement_pwm(ppm)

        # Convert PPM -> Information Content (Bits)
        # IC = 2 + sum(p * log2(p))
        ic = 2 + np.sum(ppm * np.log2(ppm + 1e-9), axis=1)
        info_matrix = ppm * ic[:, None]
        df = pd.DataFrame(info_matrix, columns=["A", "C", "G", "T"])

        # Create Logo
        # font_name='Arial Rounded MT Bold' often looks cleaner if available
        logo = logomaker.Logo(
            df, ax=ax_logo, shade_below=0, fade_below=0, color_scheme="classic"
        )

        # Styling
        ax_logo.set_ylim(0, 2)
        ax_logo.set_xlim(-0.5, df.shape[0] - 0.5)

        # Remove all borders/spines
        ax_logo.spines["top"].set_visible(False)
        ax_logo.spines["right"].set_visible(False)
        ax_logo.spines["bottom"].set_visible(True)
        ax_logo.spines["left"].set_visible(True)

        # Clean Ticks
        ax_logo.set_xticks([])  # No x-axis ticks (base positions)
        ax_logo.set_yticks([0, 2])  # Minimal y-ticks
        ax_logo.tick_params(axis="y", labelsize=6, length=2)

        # Title (TF Name)
        if strand == "-1":
            tf_name = f"{tf} (Rev. Comp.)"
        else:
            tf_name = tf
        ax_logo.set_title(tf_name, fontsize=8, pad=2, fontweight="bold")

        ax_logo.set_ylabel("Bits", fontsize=12, labelpad=1)
        ax_logo.spines["left"].set_visible(True)  # Optional:


def get_db(db_path: Path) -> gffutils.FeatureDB:
    """Open (and cache) an existing gffutils SQLite DB.

    Raises FileNotFoundError if the DB has not been built yet.
    """
    key = str(db_path)
    cached = _db_cache.get(key)
    if cached is not None:
        return cached

    if not Path(db_path).exists():
        raise FileNotFoundError(
            f"GFF DB not found at {db_path}. "
            f"Build it first with: python scripts/build_gff_db.py "
            f"--gff-dir <dir> --db-path {db_path}"
        )
    db = gffutils.FeatureDB(str(db_path))
    _db_cache[key] = db
    return db


def _get_gene_feature(
    db: gffutils.FeatureDB, gene_id: str
) -> Optional[gffutils.Feature]:
    """Look up a gene feature by ID, trying both raw and 'gene:'-prefixed
    forms (Ensembl Plants uses the prefix, others don't)."""
    try:
        return db[gene_id]
    except gffutils.FeatureNotFoundError:
        return None


def get_gene_structure(db: gffutils.FeatureDB, gene_id: str) -> dict:
    """Retrieve all sub-features of a gene with coordinates relative to the TSS.

    Args:
        db: The loaded gffutils.FeatureDB.
        gene_id: The stable gene ID (e.g., 'AT5G52310').

    Returns:
        A dictionary containing the gene's global info and a list of its
        features with both genomic and TSS-relative coordinates.
    """
    gene_id = f"gene:{gene_id}"
    gene = _get_gene_feature(db, gene_id)
    if gene is None:
        print("Gene %s not found in the GFF database.", gene_id)
        return {}

    strand = gene.strand

    # In GFF (1-based), TSS is the start for '+' strand, and end for '-' strand.
    tss = gene.start if strand == "+" else gene.end

    features = []

    # Retrieve all sub-features (exons, CDS, UTRs, mRNAs, etc.)
    for child in db.children(gene):
        # Calculate coordinates relative to the TSS.
        if strand == "+":
            rel_start = child.start - tss
            rel_end = child.end - tss
        else:
            # For '-' strand, transcription flows opposite to genomic coords.
            # The closest genomic coord to TSS (child.end) becomes rel start.
            rel_start = tss - child.end
            rel_end = tss - child.start

        features.append(
            {
                "feature_id": child.id,
                "feature_type": child.featuretype,
                "transcript_id": next(iter(child.attributes.get("Parent", [])), None),
                "genomic_start": child.start,
                "genomic_end": child.end,
                "relative_start": rel_start,
                "relative_end": rel_end,
                "attributes": dict(child.attributes),
            }
        )

    # Sort features 5' to 3' based on transcription direction
    features.sort(key=lambda x: x["relative_start"])

    return {
        "gene_id": gene_id,
        "strand": strand,
        "tss": tss,
        "genomic_start": gene.start,
        "genomic_end": gene.end,
        "features": features,
    }


FEATURE_STYLE = {
    "five_prime_UTR": ("orange", "5'UTR"),
    "three_prime_UTR": ("brown", "3'UTR"),
    "CDS": ("darkgreen", "Gene body"),
}
segments = []


def plot_ref_alt_ism(
    figure_path,
    gene,
    chrom,
    sum_lfc,
    ref_sequence,
    results_df,
    start=0,
    end=0,
    tss=0,
    strand="1",
    meme_path="",
    gff_db="",
    TSS_CENTER=1250,
    SEQ_LEN=2500,
):
    sum_lfc = sum_lfc - sum_lfc.mean(0)
    letters = ["A", "C", "G", "T"]
    letter_to_index = {l: i for i, l in enumerate(letters)}

    colors = {"A": "#2ca02c", "C": "#1f77b4", "G": "#ff7f0e", "T": "#d62728"}

    # start = max(0, min(mut_positions) - window_padding)
    # end = min(2500, max(mut_positions) + window_padding)
    region = range(start, end)
    region_length = len(region)
    mutation_dict = {}
    for _, row in results_df.iterrows():
        pos = int(row.sequence_relative_position)
        alt = row.alt
        mutation_dict.setdefault(pos, []).append(alt)
    all_mutations_pos = sorted(set(mutation_dict.keys()))

    # ---------------------------
    # Create figure
    # ---------------------------
    pwms_to_plot = list(set(map(tuple, results_df[["motif", "motif_strand"]].values)))
    fig = plt.figure(figsize=(18, 10 + ceil(len(pwms_to_plot) / 6)))
    gs = fig.add_gridspec(
        6, 1, height_ratios=[2, 1, 1, 0.5, 0.2, ceil(len(pwms_to_plot) / 6)], hspace=0.0
    )
    ax_pdf = fig.add_subplot(gs[0])
    ax_ref = fig.add_subplot(gs[1])
    ax_alt = fig.add_subplot(gs[2], sharex=ax_ref)
    ax_map = fig.add_subplot(gs[3], sharex=ax_ref)

    ax_pwms = fig.add_subplot(gs[5])

    fig.patch.set_facecolor("white")

    ax_pdf.axis("off")
    pages = convert_from_path(figure_path, dpi=300)
    ax_pdf.imshow(pages[0])
    ax_pdf.set_aspect("auto")

    for ax in [ax_ref, ax_alt]:
        ax.set_facecolor("white")

        # Remove all default spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Remove ticks completely
        ax.set_xticks([])
        ax.set_yticks([])

        # Draw custom Y axis (vertical line at left)
        ax.axvline(x=0, color="black", linewidth=1.2, ymax=0.95)

        # Draw custom X axis (horizontal zero line)
        ax.axhline(y=0, color="black", linewidth=1.0)

    ax_map.set_facecolor("white")
    for spine in ax_map.spines.values():
        spine.set_visible(False)
        # Remove ticks completely
    ax_map.set_xticks([])
    ax_map.set_yticks([])

    # ---------------------------
    # Plot reference/alternate
    # ---------------------------
    ks = 0
    for i, pos in enumerate(region):
        ref_base = ref_sequence[pos]
        idx = letter_to_index[ref_base]
        value = sum_lfc[idx, pos]
        size = get_size(all_mutations_pos, pos, region_length)
        draw_letter(ax_ref, ref_base, ks, value, colors[ref_base], size)
        if pos not in mutation_dict:
            draw_letter(ax_alt, ref_base, ks, value, colors[ref_base], size)
        else:
            for mut in list(set(mutation_dict[pos])):
                print(
                    f"A mutation at position {pos-TSS_CENTER} relative to the gene TSS, {ref_sequence[pos]} -> {mut} causes the loss of the binding site of {results_df.loc[(results_df.sequence_relative_position==pos)&(results_df.ref==ref_sequence[pos])&(results_df.alt==mut)].motif.values.tolist()}"
                )
            if len(mutation_dict[pos]) > 1:
                idx = sum_lfc[:, pos].argmin()
                alt_base = letters[idx]
            else:
                alt_base = mutation_dict[pos][0]
                idx = letter_to_index[alt_base]
            value = sum_lfc[idx, pos]
            draw_letter(ax_alt, alt_base, ks, value, colors[alt_base], size)
            x = ks
            # ---------------------------
            # Highlight mutation
            # ---------------------------
            ax_ref.axvspan(x, x + size, ymax=0.9, color="pink", alpha=0.3)
            ax_alt.axvspan(x, x + size, ymin=-0.9, color="pink", alpha=0.3)
        ks += size

    # Left edge and center x-coord for each sequence position.
    x_left = {}
    x_coords = {}
    current_x = 0
    for pos in region:
        size = get_size(all_mutations_pos, pos, region_length)
        x_left[pos] = current_x
        x_coords[pos] = current_x + (size / 2)  # center of the base
        current_x += size

    def _x_at(pos):
        """Left edge x of a sequence position; `end` maps to the far right (ks)."""
        if pos <= start:
            return 0
        if pos >= end:
            return ks
        return x_left[pos]

    # ---------------------------
    # Region map: recover feature coords from the GFF
    # ---------------------------
    def _to_seq(rel):
        """TSS-relative GFF offset -> sequence index (0..SEQ_LEN)."""
        return TSS_CENTER + rel

    segments = []  # (seq_start, seq_end, color, label)
    rel_starts = rel_ends = None

    try:
        db = get_db(Path(gff_db))
        struct = get_gene_structure(db, gene)
    except Exception as exc:
        print(f"GFF lookup failed for {gene}: {exc}")
        struct = None

    if struct and struct["features"]:
        rel_starts = [f["relative_start"] for f in struct["features"]]
        rel_ends = [f["relative_end"] for f in struct["features"]]

        # Gene body = union of all annotated features (5'UTR..3'UTR).
        gb_start = _to_seq(min(rel_starts))
        gb_end = _to_seq(max(rel_ends) + 1)
        segments.append((gb_start, gb_end, "darkgreen", "Gene body"))

        # UTRs, drawn over the gene body.
        for ftype, color, label in (
            ("five_prime_UTR", "orange", "5'UTR"),
            ("three_prime_UTR", "brown", "3'UTR"),
        ):
            matching = [f for f in struct["features"] if f["feature_type"] == ftype]
            if not matching:
                continue
            s = _to_seq(min(f["relative_start"] for f in matching))
            e = _to_seq(max(f["relative_end"] for f in matching) + 1)
            segments.append((s, e, color, label))

    # Promoter / upstream / downstream, relative to TSS and gene body.
    PROMOTER_BP = 400
    prom_start = max(0, TSS_CENTER - PROMOTER_BP)
    segments.append((prom_start, TSS_CENTER, "purple", "Promoter (400 bps)"))
    segments.append((0, prom_start, "blue", "Upstream"))
    if rel_ends is not None:
        segments.append((_to_seq(max(rel_ends) + 1), SEQ_LEN, "teal", "Downstream"))

    # ---------------------------
    # TSS arrow
    # ---------------------------
    if start <= TSS_CENTER < end:
        tss_x = x_coords[TSS_CENTER]
        ax_map.annotate(
            "",
            xy=(tss_x, 0.25),
            xytext=(tss_x, -0.05),
            arrowprops=dict(arrowstyle="->", color="black", lw=2),
        )
        ax_map.text(
            tss_x,
            0.3,
            "TSS",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", pad=1),
        )

    # ---------------------------
    # Draw only the segments visible in [start, end),
    # stacking labels onto separate rows when they'd overlap.
    # ---------------------------
    label_rows = []  # list of lists of (x0, x1) occupied per row

    def _pick_row(lx0, lx1):
        for r, spans in enumerate(label_rows):
            if all(lx1 < s0 or lx0 > s1 for s0, s1 in spans):
                spans.append((lx0, lx1))
                return r
        label_rows.append([(lx0, lx1)])
        return len(label_rows) - 1

    half_w = ks * 0.06  # assumed half-width of a label in x-units; tune if needed

    for seg_start, seg_end, color, label in segments:
        vis_start = max(seg_start, start)
        vis_end = min(seg_end, end)
        if vis_end <= vis_start:
            continue  # feature not in the plotted window

        x0 = _x_at(vis_start)
        x1 = _x_at(vis_end)
        if x1 <= x0:
            continue

        # Genomic coords of the visible span, for the label.
        g0 = relative_to_genomic_position(vis_start, tss, strand, TSS_CENTER)
        g1 = relative_to_genomic_position(vis_end - 1, tss, strand, TSS_CENTER)

        ax_map.hlines(0, x0, x1, colors=color, lw=4)

        cx = (x0 + x1) / 2
        text = f"{label} {min(g0, g1)}–{max(g0, g1)}"
        row = _pick_row(cx - half_w, cx + half_w)
        ax_map.text(
            cx,
            -0.08 - row * 0.32,
            text,
            color=color,
            ha="center",
            va="top",
            fontsize=8,
        )

    # ---------------------------
    # Scale Y consistently
    # ---------------------------
    sub_scores = sum_lfc[:, start:end]
    ymin = np.min(sub_scores)
    ymax = np.max(sub_scores)

    pad = 0.05 * (ymax - ymin)

    ax_ref.set_ylim(0.75 * ymin, ymax + pad)
    ax_alt.set_ylim(0.75 * ymin, ymax + pad)
    ax_ref.set_xlim(-0.5, ks + 0.5)
    ax_alt.set_xlim(-0.5, ks + 0.5)
    ax_map.set_ylim(-1.2, 0.5)

    # ---------------------------
    # Titles and annotations
    # ---------------------------

    # Left labels
    ax_ref.text(
        0.01,
        0.85,
        "Reference allele",
        transform=ax_ref.transAxes,
        fontsize=13,
        ha="left",
    )
    ax_alt.text(
        0.01,
        0.85,
        "Alternative allele",
        transform=ax_alt.transAxes,
        fontsize=13,
        ha="left",
    )

    # Variant text (top right)
    variant_text = f"{gene} ({chrom}: {relative_to_genomic_position(start,tss,strand, TSS_CENTER)} > {relative_to_genomic_position(end,tss,strand, TSS_CENTER)}), strand {strand}"
    ax_ref.text(
        0.99, 0.85, variant_text, transform=ax_ref.transAxes, fontsize=13, ha="right"
    )
    ax_alt.text(0.99, 0.85, "ISM", transform=ax_alt.transAxes, fontsize=13, ha="right")
    plot_pwm(
        ax_pwms,
        meme_path,
        pwms_to_plot,
    )

    plt.tight_layout()
    plt.show()


def create_zoom_in_bed(deep_plant_path: str, chrom: str | int, start: int, end: int):
    """
    Creates a Zoom_in.bed file for pyGenomeTracks with specific start and end markers.
    """
    directory = os.path.join(deep_plant_path, "results/pygenometracks/")
    file_path = os.path.join(directory, "Zoom_in.bed")
    create_path(path=directory)
    line1 = f"Chr{chrom}\t{start}\t{start+1}\tstart\n"
    line2 = f"Chr{chrom}\t{end}\t{end+1}\tend\n"

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines([line1, line2])

    print(f"Successfully created Zoom_in.bed at: {file_path}")
    return file_path


def create_track_ini(deep_plant_path: str, mutations_bed_path: str):
    """
    Creates a track.ini file for pyGenomeTracks at the specified deep_plant_path.
    """
    # Define the target directory and full file path
    directory = os.path.join(deep_plant_path, "results/pygenometracks/")
    file_path = os.path.join(directory, "track.ini")

    # Ensure the target directory exists
    create_path(path=directory)

    # Define the INI content, evaluating the deep_plant_path variable into the paths
    content = f"""[test bigwig1]
file = {deep_plant_path}/data/arabidopsis/bw/SRX111004_Rep0.rpgc.bw
height = 1
title = OC(WP)
min_value = 0
max_value = 10

[test bigwig2]
file = {deep_plant_path}/data/arabidopsis/bw/SRX111007_Rep0.rpgc.bw
height = 1
title = OC(Flower)
min_value = 0
max_value = 10

[test bigwig3]
file = {deep_plant_path}/data/arabidopsis/bw/SRX3041700_Rep0.rpgc.bw
height = 1
title = OC(Leaf)
min_value = 0
max_value = 30

[test bigwig4]
file = {deep_plant_path}/data/arabidopsis/bw/SRX5036313_Rep0.rpgc.bw
height = 1
title = OC(Root)
min_value = 0
max_value = 10

[test bigwig5]
file = {deep_plant_path}/data/arabidopsis/bw/SRX7780559_Rep0.rpgc.bw
height = 1
title = OC(Cold)
min_value = 0
max_value = 5

[spacer]
height = 0.5

[genes]
file = {deep_plant_path}/data/arabidopsis/gtf/Arabidopsis.gtf
height = 4
style = flybase
display = stacked
color = #2F5D50
color_arrow = #1F3D36
arrow_interval = 50
fontsize = 12
labels = true
prefered_name = gene_id
merge_transcripts = true

[vlines]
file = {mutations_bed_path}
type = vlines
color = #C00000
alpha = 0.6
line_width = 2

[x-axis]
fontsize=12
"""

    # Write the content to the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Successfully created track.ini at: {file_path}")
    return file_path
