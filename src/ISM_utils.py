import math
import os
import subprocess
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Optional

import gffutils
import logomaker
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from Bio import motifs
from Bio.motifs.matrix import FrequencyPositionMatrix
from fastprogress import progress_bar
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import Patch, PathPatch
from matplotlib.textpath import TextPath
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Affine2D, offset_copy
from pdf2image import convert_from_path

from src.utils import create_path

NUCLEOTIDES = ["A", "C", "G", "T"]
NUC_TO_IDX = {n: i for i, n in enumerate(NUCLEOTIDES)}


# ======================================================================
# 1. Motif database
# ======================================================================


def load_meme_database(meme_path: str, format: Optional[str] = "pfm"):
    """Load a MEME v4 text file.

    format="ppm": {motif_name: probability matrix, shape (4, w)}
    format="pfm": {motif_name: Biopython Motif with counts = probs * nsites}
    """
    motifs_dict = {}
    with open(meme_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    i = 0
    while i < len(lines):
        if lines[i].startswith("MOTIF"):
            parts = lines[i].split()
            motif_id = parts[1]
            motif_name = parts[2] if len(parts) > 2 else motif_id

            i += 1
            header = lines[i]
            if not header.startswith("letter-probability matrix"):
                raise ValueError(f"Expected matrix header after MOTIF {motif_id}")
            w = int(header.split("w=")[1].split()[0])
            nsites = float(header.split("nsites=")[1].split()[0])

            probs = np.array(
                [[float(x) for x in lines[i + k + 1].split()] for k in range(w)]
            ).T  # (4, w)
            i += w

            if format == "ppm":
                motifs_dict[motif_name] = probs
            elif format == "pfm":
                counts = probs * nsites
                m = motifs.Motif(alphabet="ACGT")
                m.counts = FrequencyPositionMatrix(
                    m.alphabet, {n: counts[j] for j, n in enumerate(NUCLEOTIDES)}
                )
                m.id, m.name = motif_id, motif_name
                motifs_dict[motif_name] = m
        i += 1
    return motifs_dict


def get_latest_motifs(motif_dict):
    """Keep only the newest version of each JASPAR base ID ('MA1186.2.TF' > '.1')."""
    latest = {}
    for name in motif_dict:
        base_id, version = name.split(".")[0], int(name.split(".")[1])
        if base_id not in latest or version > latest[base_id][0]:
            latest[base_id] = (version, name)
    return {name: motif_dict[name] for _, name in latest.values()}


def nuclt_dict(sequences):
    """Background nucleotide frequencies over a list of sequences."""
    counts = {n: sum(s.count(n) for s in sequences) for n in NUCLEOTIDES}
    total = sum(counts.values())
    return {n: c / total for n, c in counts.items()}


def get_motifs(motifs_dict, background):
    """PFM motifs -> PSSMs (pseudocount 0.5, log-odds vs background)."""
    return {
        tf: m.counts.normalize(pseudocounts=0.5).log_odds(background=background)
        for tf, m in motifs_dict.items()
    }


def ppm_to_pwm(ppm, background):
    bg = np.array([background[n] for n in NUCLEOTIDES])
    return np.log2(ppm / bg)


def reverse_complement_pwm(ppm):
    """(L, 4) matrix -> reverse complement (reverse rows, swap A<->T, C<->G)."""
    return ppm[::-1, [3, 2, 1, 0]]


def get_motif_name(matrix_id):
    """JASPAR API name for a matrix ID (e.g. 'MA1192.2'); returns the ID on failure."""
    try:
        r = requests.get(
            f"https://jaspar.elixir.no/api/v1/matrix/{matrix_id}", timeout=20
        )
        r.raise_for_status()
        return r.json()["name"]
    except Exception:
        return matrix_id


# ======================================================================
# 2. Coordinates / GFF
# ======================================================================


def relative_to_genomic_position(rel_pos, tss, strand, center=1250):
    """Index in the TSS-centred input sequence -> genomic coordinate.
    strand is 1 / -1 (int or str)."""
    return int(tss) + (int(rel_pos) - center) * int(strand)


def extract_chromosome(header):
    return next((t.strip() for t in header.split() if "Chr" in t), None)


_db_cache: dict[str, gffutils.FeatureDB] = {}  # keyed by DB path


def get_db(db_path: Path) -> gffutils.FeatureDB:
    """Open (and cache) an existing gffutils SQLite DB."""
    key = str(db_path)
    if key in _db_cache:
        return _db_cache[key]
    if not Path(db_path).exists():
        raise FileNotFoundError(
            f"GFF DB not found at {db_path}. Build it first with: "
            f"python scripts/build_gff_db.py --gff-dir <dir> --db-path {db_path}"
        )
    _db_cache[key] = gffutils.FeatureDB(key)
    return _db_cache[key]


def _get_gene_feature(db, gene_id) -> Optional[gffutils.Feature]:
    """Look up a gene by 'gene:<ID>' (Ensembl Plants) or the bare ID."""
    bare = gene_id.removeprefix("gene:")
    for key in (f"gene:{bare}", bare):
        try:
            return db[key]
        except gffutils.FeatureNotFoundError:
            continue
    return None


def get_gene_structure(db, gene_id: str) -> dict:
    """All sub-features of a gene with genomic and TSS-relative coordinates,
    sorted 5'->3' along the direction of transcription."""
    gene = _get_gene_feature(db, gene_id)
    if gene is None:
        print(f"Gene {gene_id} not found in the GFF database.")
        return {}

    strand = gene.strand
    tss = gene.start if strand == "+" else gene.end  # GFF is 1-based
    features = []
    for child in db.children(gene):
        if strand == "+":
            rel_start, rel_end = child.start - tss, child.end - tss
        else:
            rel_start, rel_end = tss - child.end, tss - child.start
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
    features.sort(key=lambda f: f["relative_start"])
    return {
        "gene_id": gene.id,
        "strand": strand,
        "tss": tss,
        "genomic_start": gene.start,
        "genomic_end": gene.end,
        "features": features,
    }


# ======================================================================
# 3. TF aliases / ChIP peaks
# ======================================================================


@lru_cache(maxsize=None)  # one MyGene query per TF per session
def get_tf_info(query_term):
    """Arabidopsis aliases for a TF via MyGene (symbol, aliases, AGI locus)."""
    url = (
        f"https://mygene.info/v3/query?q={query_term}&species=3702"
        f"&fields=symbol,name,alias,locus_tag"
    )
    try:
        r = requests.get(url, timeout=10)
        hits = r.json().get("hits", []) if r.status_code == 200 else []
    except Exception:
        hits = []
    if not hits:
        return (query_term,)
    h = hits[0]
    aliases = h.get("alias", [])
    aliases = [aliases] if isinstance(aliases, str) else list(aliases)
    aliases += [h.get("symbol", ""), h.get("locus_tag", ""), query_term]
    return tuple(set(a for a in aliases if a))


def _dedupe(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def get_tf_info_ensembl(query_term):
    """Alternative alias lookup via Ensembl REST."""
    try:
        headers = {"Content-Type": "application/json"}
        r = requests.get(
            f"https://rest.ensembl.org/xrefs/symbol/arabidopsis_thaliana/{query_term}?",
            headers=headers,
            timeout=20,
        )
        data = r.json() if r.status_code == 200 else []
        gene_id = data[0].get("id") if data else None
        if gene_id is None:
            return [query_term]
        r2 = requests.get(
            f"https://rest.ensembl.org/lookup/id/{gene_id}?",
            headers=headers,
            timeout=20,
        )
        if r2.status_code != 200:
            return [query_term]
        info = r2.json()
        return _dedupe(
            [info.get("display_name", "Unknown"), info.get("id", "Unknown"), query_term]
        )
    except Exception:
        return [query_term]


def get_tf_info_uniprot(query_term):
    """Alternative alias lookup via UniProt."""
    try:
        url = (
            f"https://rest.uniprot.org/uniprotkb/search?"
            f"query={query_term} AND organism_id:3702&format=json&size=1"
        )
        r = requests.get(url, timeout=20)
        results = r.json().get("results", []) if r.status_code == 200 else []
        if not results:
            return [query_term]
        entry = results[0]
        aliases = []
        for g in entry.get("genes", []):
            if "geneName" in g:
                aliases.append(g["geneName"]["value"])
            aliases += [s["value"] for s in g.get("synonyms", [])]
        aliases += [entry.get("primaryAccession", "Unknown"), query_term]
        return _dedupe(aliases)
    except Exception:
        return [query_term]


@lru_cache(maxsize=None)
def _cached_read_bed(filepath):
    """Each BED file is read from disk once per session."""
    return pd.read_csv(filepath, delimiter="\t", header=None)


def load_bed_for_tf(tf_name, metadata, bed_files, bed_files_path):
    """IDR peak BED for a TF, preferring WT / untreated / Col-0 samples.
    Returns a DataFrame, or None if no file is available."""
    tf_rows = metadata[metadata.Factor.isin(get_tf_info(tf_name))]
    if tf_rows.empty:
        return None

    priorities = [
        (tf_rows.Mutant == "WT") & (tf_rows.Treatment == "No treatment"),
        tf_rows.Mutant == "WT",
        tf_rows.Treatment == "No treatment",
        tf_rows.Type_Strain == "Col-0",
        pd.Series(True, index=tf_rows.index),  # any sample of this TF
    ]
    for cond in priorities:
        subset = tf_rows[cond]
        if subset.empty:
            continue
        row = subset.iloc[0]
        for stem in (row["Sample accession number"], row["Experiment Name"]):
            fname = f"{stem}.target.idr.bed"
            if fname in bed_files:
                return _cached_read_bed(os.path.join(bed_files_path, fname))
    return None


def check_peak_overlap(bed_df, chromosome, genomic_pos, window=0):
    hits = (
        (bed_df[0] == chromosome)
        & (bed_df[1] <= genomic_pos + window)
        & (bed_df[2] >= genomic_pos - window)
    )
    return bool(hits.any())


# ======================================================================
# 4. Motif scanning
# ======================================================================


def _first_hit(sequence, pssm, rel_thresh):
    """(position, score) of the first PSSM hit above rel_thresh * max score,
    or (-1, -inf). Negative positions are reverse-strand hits (Biopython)."""
    if pssm.length > len(sequence):
        return -1, -float("inf")
    for pos, score in pssm.search(sequence, threshold=pssm.max * rel_thresh):
        return pos, score
    return -1, -float("inf")


def scan_variant_effects_from_dict(ref_seq, alt_seq, motif_dict, score_threshold=0.0):
    """Motifs whose binding site is lost or gained by the ref -> alt change.

    A site counts as present above score_threshold * max score and absent
    below 0.5 * max score. Returns [(motif_name, 'Loss'|'Gain', hit_pos), ...].
    """
    if ref_seq == alt_seq:
        return []

    affected = []
    for name, pssm in motif_dict.items():
        pos_ref, score_ref = _first_hit(ref_seq, pssm, score_threshold)
        pos_alt, score_alt = _first_hit(alt_seq, pssm, score_threshold)
        hit_cut, absent_cut = pssm.max * score_threshold, pssm.max * 0.5

        if score_ref >= hit_cut:
            if score_alt < absent_cut:
                affected.append((name, "Loss", pos_ref))
        elif score_alt >= hit_cut and score_ref < absent_cut:
            affected.append((name, "Gain", pos_alt))
    return affected


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
    """Scan high-impact ISM positions for motif gain/loss and ChIP support.

    sum_lfc : (4, L) ISM for one experiment, already multiplied by the
              regulation sign (1 = activators, -1 = repressors).
    """
    center = seq_len // 2
    ref_idx = np.array([NUC_TO_IDX[n] for n in ref_sequence])
    ism_ref = (sum_lfc - sum_lfc.mean(0))[ref_idx, np.arange(len(ref_sequence))]

    row = gene_df_full.loc[gene_df_full.gene == gene].iloc[0]
    strand, tss, chromosome = str(row.strand), int(row.tss), f"Chr{row.chrom}"
    bed_files = frozenset(os.listdir(bed_files_path))

    results = []
    for pos in progress_bar(np.where(ism_ref > mutation_threshold * ism_ref.max())[0]):
        mutation = nuclt[np.argmin(sum_lfc[:, pos])]
        if ref_sequence[pos] == mutation:
            continue

        lo = max(0, pos - pad)
        ref_local = ref_sequence[lo : pos + pad]
        alt_local = ref_sequence[lo:pos] + mutation + ref_sequence[pos + 1 : pos + pad]
        affected = scan_variant_effects_from_dict(
            ref_local, alt_local, motif_dict, score_threshold
        )
        if not affected:
            continue

        genomic_pos = relative_to_genomic_position(pos, tss, strand, center)
        for motif_name, effect, hit_pos in affected:
            bed_df = load_bed_for_tf(
                motif_name.split(".")[2], metadata, bed_files, bed_files_path
            )
            peak_status = (
                "No bed file available"
                if bed_df is None
                else check_peak_overlap(bed_df, chromosome, genomic_pos)
            )
            results.append(
                {
                    "gene": gene,
                    "gene_strand": strand,
                    "chromosome": chromosome,
                    "tss_relative_position": pos - center,
                    "gene_region": "promoter" if pos < center else "downstream",
                    "sequence_relative_position": pos,
                    "genomic_position": genomic_pos,
                    "ref": ref_sequence[pos],
                    "alt": mutation,
                    "motif": motif_name,
                    "motif_strand": "1" if hit_pos >= 0 else "-1",
                    "effect": effect,
                    "chip_peak_overlap": peak_status,
                }
            )
    return pd.DataFrame(results)


# ======================================================================
# 5. Plot style and drawing helpers
# ======================================================================
# Change fonts / line width after import if needed:
#     import src.ISM_utils as iu
#     iu.FS.update(tiny=5, base=6); iu.set_linewidth(0.6)

FS = {"tiny": 4, "base": 5, "label": 6, "panel": 7}  # pt
LW = 0.5


def set_linewidth(lw):
    global LW
    LW = lw


UP_COLOR = "#1f77b4"  # activators (blue)
DOWN_COLOR = "#d62728"  # repressors (red)
BOTH_COLOR = "purple"  # position flagged in both directions
REG_LABELS = {1: "Activators", -1: "Repressors"}
REG_COLORS = {1: UP_COLOR, -1: DOWN_COLOR}
BASE_COLORS = {"A": "#2ca02c", "C": "#1f77b4", "G": "#ff7f0e", "T": "#d62728"}


def get_size(mutation_pos, pos, region_length):
    """Letter width: full size near flagged positions, shrunk elsewhere."""
    if any(abs(pos - p) < 5 for p in mutation_pos):
        return 1.0
    if region_length < 500:
        return 0.25
    if region_length < 1250:
        return 0.1
    return 0.05


def draw_letter(ax, letter, x, height, color, size):
    """Nucleotide glyph at x; negative height draws it head-down."""
    if height == 0:
        return
    text = TextPath(
        (0, 0),
        letter,
        size=size,
        prop=fm.FontProperties(family="DejaVu Sans", weight="bold"),
    )
    transform = Affine2D().scale(1, height).translate(x, 0)
    ax.add_patch(PathPatch(text, lw=0, fc=color, transform=transform + ax.transData))


def _style_logo_axis(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axvline(x=0, color="black", linewidth=LW, ymax=0.95)
    ax.axhline(y=0, color="black", linewidth=LW)


def _set_logo_limits(ax, vals, ks):
    vals = np.asarray(vals)
    lo, hi = min(vals.min(), 0.0), max(vals.max(), 0.0)
    pad = 0.08 * (hi - lo if hi > lo else 1.0)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(-0.5, ks + 0.5)


def _highlight(ax, x, width, color):
    ax.axvspan(
        x,
        x + width,
        ymax=1.0,
        facecolor=color,
        edgecolor="none",
        linewidth=0,
        alpha=0.2,
    )


def _draw_ref_track(
    ax,
    sum_lfc,
    ref_sequence,
    size_positions,
    up_positions,
    down_positions,
    start,
    end,
    region_length,
):
    """Reference-allele ISM logo with direction-coloured highlights.
    Returns (total x-width, {seq_pos: left x})."""
    sum_lfc = sum_lfc - sum_lfc.mean(0)
    _style_logo_axis(ax)

    ks, x_left, vals = 0, {}, []
    for pos in range(start, end):
        size = get_size(size_positions, pos, region_length)
        x_left[pos] = ks
        base = ref_sequence[pos]
        val = sum_lfc[NUC_TO_IDX[base], pos]
        vals.append(val)
        draw_letter(ax, base, ks, val, BASE_COLORS[base], size)

        in_up, in_down = pos in up_positions, pos in down_positions
        if in_up or in_down:
            _highlight(
                ax,
                ks,
                size,
                BOTH_COLOR if in_up and in_down else UP_COLOR if in_up else DOWN_COLOR,
            )
        ks += size

    _set_logo_limits(ax, vals, ks)
    return ks, x_left


def _draw_alt_track(
    ax, sum_lfc, ref_sequence, size_positions, alt_by_pos, start, end, region_length
):
    """Alternative-allele ISM logo: alt base at flagged positions, ref elsewhere."""
    sum_lfc = sum_lfc - sum_lfc.mean(0)
    _style_logo_axis(ax)

    ks, vals = 0, []
    for pos in range(start, end):
        size = get_size(size_positions, pos, region_length)
        if pos in alt_by_pos:
            base, color = alt_by_pos[pos]
            _highlight(ax, ks, size, color)
        else:
            base = ref_sequence[pos]
        val = sum_lfc[NUC_TO_IDX[base], pos]
        vals.append(val)
        draw_letter(ax, base, ks, val, BASE_COLORS[base], size)
        ks += size

    _set_logo_limits(ax, vals, ks)


def _add_y_scale(ax, labelsize=None):
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.tick_params(
        axis="y", labelsize=labelsize or FS["tiny"], length=2, width=LW, pad=1
    )


def _draw_annotation_map(
    ax,
    gene,
    start,
    end,
    x_left,
    ks,
    gff_db,
    TSS_CENTER=1250,
    SEQ_LEN=2500,
    promoter_bp=400,
):
    """Upstream / promoter / 5'UTR / gene body / 3'UTR / downstream + TSS arrow."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    def x_at(pos):
        if pos <= start:
            return 0
        if pos >= end:
            return ks
        return x_left.get(pos, ks)

    def to_seq(rel):
        return TSS_CENTER + rel

    try:
        struct = get_gene_structure(get_db(Path(gff_db)), gene)
    except Exception as exc:
        print(f"GFF lookup failed for {gene}: {exc}")
        struct = None

    segments, rel_ends = [], None
    if struct and struct["features"]:
        feats = struct["features"]
        rel_ends = [f["relative_end"] for f in feats]
        segments.append(
            (
                to_seq(min(f["relative_start"] for f in feats)),
                to_seq(max(rel_ends) + 1),
                "darkgreen",
                "Gene body",
            )
        )
        for ftype, color, label in (
            ("five_prime_UTR", "orange", "5'UTR"),
            ("three_prime_UTR", "brown", "3'UTR"),
        ):
            match = [f for f in feats if f["feature_type"] == ftype]
            if match:
                segments.append(
                    (
                        to_seq(min(f["relative_start"] for f in match)),
                        to_seq(max(f["relative_end"] for f in match) + 1),
                        color,
                        label,
                    )
                )

    prom_start = max(0, TSS_CENTER - promoter_bp)
    segments.append((prom_start, TSS_CENTER, "purple", "Promoter"))
    segments.append((0, prom_start, "blue", "Upstream"))
    if rel_ends is not None:
        segments.append((to_seq(max(rel_ends) + 1), SEQ_LEN, "teal", "Downstream"))

    if start <= TSS_CENTER < end:
        tss_x = x_at(TSS_CENTER)
        ax.annotate(
            "",
            xy=(tss_x, 0.25),
            xytext=(tss_x, -0.05),
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        )
        ax.text(
            tss_x,
            0.32,
            "TSS",
            ha="center",
            va="bottom",
            fontsize=FS["base"],
            fontweight="bold",
        )

    label_rows = []  # occupied x-spans per label row

    def pick_row(a, b):
        for r, spans in enumerate(label_rows):
            if all(b < s0 or a > s1 for s0, s1 in spans):
                spans.append((a, b))
                return r
        label_rows.append([(a, b)])
        return len(label_rows) - 1

    half_w = ks * 0.06  # approximate half-width of a label, in x units
    for s_seg, e_seg, color, label in segments:
        vs, ve = max(s_seg, start), min(e_seg, end)
        if ve <= vs:
            continue
        x0, x1 = x_at(vs), x_at(ve)
        if x1 <= x0:
            continue
        ax.hlines(0, x0, x1, colors=color, lw=2)
        cx = (x0 + x1) / 2
        row = pick_row(cx - half_w, cx + half_w)
        ax.text(
            cx,
            -0.10 - row * 0.30,
            label,
            color=color,
            ha="center",
            va="top",
            fontsize=FS["tiny"],
        )

    ax.set_xlim(-0.5, ks + 0.5)
    ax.set_ylim(-1.2, 0.5)


def _ppm_to_bits(ppm):
    """(L, 4) probability matrix -> information content (bits)."""
    ppm = np.clip(ppm, 1e-9, 1)
    ppm = ppm / ppm.sum(1, keepdims=True)
    ic = 2 + np.sum(ppm * np.log2(ppm), axis=1)
    return ppm * ic[:, None]


def _draw_logo(ax, ppm, title, show_y, fontsize=None):
    df = pd.DataFrame(_ppm_to_bits(ppm), columns=NUCLEOTIDES)
    logomaker.Logo(df, ax=ax, shade_below=0, fade_below=0, color_scheme="classic")
    ax.set_ylim(0, 2)
    ax.set_xlim(-0.5, len(df) - 0.5)
    ax.set_xticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(LW)
    if show_y:
        ax.set_yticks([0, 2])
        ax.tick_params(axis="y", labelsize=FS["tiny"], length=2, width=LW, pad=1)
        ax.set_ylabel("Bits", fontsize=FS["tiny"], labelpad=1)
        ax.spines["left"].set_linewidth(LW)
    else:
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    ax.set_title(title, fontsize=fontsize or FS["base"], pad=1.5, fontweight="bold")


def _underlined_label(ax, x, y, text, ha, color, fontsize, renderer):
    txt = ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va="center",
        fontsize=fontsize,
        color=color,
    )
    bb = txt.get_window_extent(renderer).transformed(ax.transAxes.inverted())
    ax.plot(
        [bb.x0, bb.x1],
        [bb.y0 - 0.05] * 2,
        transform=ax.transAxes,
        color=color,
        lw=LW,
        clip_on=False,
    )


def _panel_letter(fig, spec, letter, dx=-10, dy=4, specific_dx=-20):
    if letter in ("A", "B"):
        dx = specific_dx
    pos = spec.get_position(fig)
    tr = offset_copy(fig.transFigure, fig=fig, x=dx, y=dy, units="points")
    fig.text(
        pos.x0,
        pos.y1,
        letter,
        transform=tr,
        fontsize=FS["panel"],
        fontweight="bold",
        ha="right",
        va="bottom",
    )


def _trim_whitespace(img, thresh=245, pad=2):
    """Crop near-white margins from a rendered page."""
    arr = np.asarray(img)
    gray = arr[..., :3].mean(-1) if arr.ndim == 3 else arr
    rows = np.where((gray < thresh).any(1))[0]
    cols = np.where((gray < thresh).any(0))[0]
    if rows.size == 0:
        return arr
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad + 1, arr.shape[0])
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad + 1, arr.shape[1])
    return arr[r0:r1, c0:c1]


def _resolve_motif(name, ppms):
    """MEME key for a TF name / JASPAR ID / full key (newest version if several)."""
    if name in ppms:
        return name
    q = name.upper()

    def matches(k):
        parts = k.upper().split(".")
        return q in (parts[-1], parts[0], ".".join(parts[:2]))

    def version(k):
        parts = k.split(".")
        return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

    cands = [k for k in ppms if matches(k)]
    return max(cands, key=version) if cands else None


def _resolve_pwm_groups(meme_path, groups):
    """{group: [(name, strand[, label]), ...]} -> [(group, [(label, ppm(L,4)), ...])]."""
    ppms = load_meme_database(meme_path, format="ppm")
    rows = []
    for group, entries in OrderedDict(groups).items():
        items = []
        for entry in entries:
            name, strand = entry[0], str(entry[1])
            label = entry[2] if len(entry) > 2 else None
            key = _resolve_motif(name, ppms)
            if key is None:
                print(f"[PWM] '{name}' not found in {meme_path}; skipped")
                continue
            ppm = np.asarray(ppms[key]).T
            tf = key.split(".")[-1].upper()
            if strand in ("-", "-1"):
                ppm = reverse_complement_pwm(ppm)
                label = label or f"{tf} (rc)"
            items.append((label or tf, ppm))
        if items:
            rows.append((group, items))
    return rows


# ======================================================================
# 6. pyGenomeTracks
# ======================================================================

OC_TRACKS = [  # (SRX accession, title, max_value)
    ("SRX111007", "OC(Flower)", "auto"),
    ("SRX3041700", "OC(Leaf)", "auto"),
    ("SRX5036313", "OC(Root)", "auto"),
    ("SRX7780553", "OC(Whole Plant)", "auto"),
    ("SRX7780559", "OC(WP-Cold)", "auto"),
    ("SRX7780579", "OC(WP-Drought)", "auto"),
    ("SRX7780565", "OC(WP-Heat)", "auto"),
    ("SRX7780572", "OC(WP-Salt)", "auto"),
]


def _pgt_dir(deep_plant_path):
    directory = os.path.join(deep_plant_path, "results/pygenometracks/")
    create_path(path=directory)
    return directory


def create_zoom_in_bed(deep_plant_path: str, chrom, start: int, end: int):
    """BED with two 1-bp markers (start, end) drawn as vlines by pyGenomeTracks."""
    file_path = os.path.join(_pgt_dir(deep_plant_path), "Zoom_in.bed")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(
            f"Chr{chrom}\t{start}\t{start + 1}\tstart\n"
            f"Chr{chrom}\t{end}\t{end + 1}\tend\n"
        )
    return file_path


def create_track_ini(deep_plant_path, mutations_bed_path, fontsize=6, tracks=OC_TRACKS):
    """track.ini with the open-chromatin bigwigs, gene models and vlines."""
    file_path = os.path.join(_pgt_dir(deep_plant_path), "track_single_gene.ini")
    bw = f"{deep_plant_path}/data/arabidopsis/bw"
    blocks = [
        f"[bigwig{i}]\nfile = {bw}/{srx}_Rep0.rpgc.bw\nheight = 1\n"
        f"title = {title}\nmin_value = 0\nmax_value = {vmax}\n"
        for i, (srx, title, vmax) in enumerate(tracks, 1)
    ]
    content = "\n".join(blocks) + f"""
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
fontsize = {fontsize}
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
fontsize = {fontsize}
"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return file_path


def _make_gene_genometracks_pdf(
    deep_plant_path,
    gene,
    exp_id,
    chrom,
    tss,
    strand,
    start,
    end,
    center,
    flank=5000,
    fontsize=6,
    width_cm=18.3,
):
    """Render the track panel (TSS ± flank); vlines mark the ISM window edges."""
    g0 = relative_to_genomic_position(start, tss, strand, center)
    g1 = relative_to_genomic_position(end - 1, tss, strand, center)
    bed = create_zoom_in_bed(deep_plant_path, chrom, g0, g1)
    ini = create_track_ini(deep_plant_path, bed, fontsize=fontsize)

    out_pdf = os.path.join(_pgt_dir(deep_plant_path), f"track_{gene}_{exp_id}.pdf")
    if os.path.exists(out_pdf):
        os.remove(out_pdf)  # never reuse an old render
    subprocess.run(
        [
            "pyGenomeTracks",
            "--tracks",
            ini,
            "--region",
            f"Chr{chrom}:{max(1, tss - flank)}-{tss + flank}",
            "--outFileName",
            out_pdf,
            "--dpi",
            "500",
            "--trackLabelFraction",
            "0.04",
            "--width",
            str(width_cm),
            "--fontSize",
            str(fontsize),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return out_pdf


# ======================================================================
# 7. Regulator filtering
# ======================================================================


def _filtered_df(genes_dfs, gene, exp_id, reg, chip_filter=True):
    """Scan results for one direction; with chip_filter, Loss + ChIP-supported only."""
    df = genes_dfs.get(f"{gene}_{exp_id}_{reg}")
    if df is None or df.empty:
        return None
    if chip_filter:
        df = df[
            (df["effect"] == "Loss") & (df["chip_peak_overlap"].astype(str) == "True")
        ]
    return df


def _paired_motif_rows(df, max_dist=5):
    """Rows whose motif is disrupted at >= 2 positions within max_dist bp."""
    if df is None or df.empty:
        return pd.DataFrame()
    kept = []
    for _, g in df.groupby("motif"):
        p = np.sort(g["sequence_relative_position"].astype(int).unique())
        if len(p) < 2:
            continue
        close = np.diff(p) <= max_dist
        mask = np.zeros(len(p), dtype=bool)
        mask[1:] |= close  # near the previous position
        mask[:-1] |= close  # near the next position
        if mask.any():
            kept.append(g[g["sequence_relative_position"].astype(int).isin(p[mask])])
    return pd.concat(kept) if kept else pd.DataFrame()


def regulator_table(genes_dfs, gene, exp_id, chip_filter=True, max_dist=5):
    """Paired, ChIP-supported Loss mutations in both directions, with a
    'direction' column ('Activators' / 'Repressors')."""
    parts = []
    for reg in (1, -1):
        df = _paired_motif_rows(
            _filtered_df(genes_dfs, gene, exp_id, reg, chip_filter), max_dist
        )
        if not df.empty:
            parts.append(df.assign(direction=REG_LABELS[reg]))
    if not parts:
        return pd.DataFrame()
    return (
        pd.concat(parts)
        .sort_values(["direction", "motif", "sequence_relative_position"])
        .reset_index(drop=True)
    )


def _alt_bases(table):
    """{seq_pos: (alt_base, highlight colour)} for the alt-allele track."""
    out = {}
    for reg in (1, -1):
        sub = table[table.direction == REG_LABELS[reg]]
        for pos, g in sub.groupby("sequence_relative_position"):
            pos = int(pos)
            out[pos] = (g["alt"].iloc[0], BOTH_COLOR if pos in out else REG_COLORS[reg])
    return out


def _regulator_pwm_groups(table):
    """{'Activators': [(motif, strand), ...], 'Repressors': [...]}, ordered 5'->3'."""
    groups = OrderedDict()
    for reg in (1, -1):
        sub = table[table.direction == REG_LABELS[reg]]
        if sub.empty:
            continue
        sub = sub.sort_values("sequence_relative_position")
        groups[REG_LABELS[reg]] = list(
            OrderedDict.fromkeys(zip(sub["motif"], sub["motif_strand"].astype(str)))
        )
    return groups


def _plot_regulator_logos(
    fig, spec, meme_path, groups, n_cols=6, wspace=0.45, hspace=0.9
):
    """Logos grouped by direction; groups longer than n_cols wrap to new rows."""
    rows = _resolve_pwm_groups(meme_path, groups) if groups else []
    if not rows:
        fig.add_subplot(spec).axis("off")
        return

    layout = [
        (g, items[i : i + n_cols], i == 0)
        for g, items in rows
        for i in range(0, len(items), n_cols)
    ]
    colors = {REG_LABELS[r]: REG_COLORS[r] for r in (1, -1)}
    gs = spec.subgridspec(
        len(layout), 2, width_ratios=[0.14, 1], wspace=0.06, hspace=hspace
    )
    renderer = fig.canvas.get_renderer()

    for r, (g, chunk, first_row) in enumerate(layout):
        ax_lab = fig.add_subplot(gs[r, 0])
        ax_lab.axis("off")
        if first_row:
            _underlined_label(
                ax_lab, 0.80, 0.5, g, "right", colors[g], FS["label"], renderer
            )
        gs_row = gs[r, 1].subgridspec(1, n_cols, wspace=wspace)
        for c, (label, ppm) in enumerate(chunk):
            ax = fig.add_subplot(gs_row[c])
            _draw_logo(ax, ppm, label, show_y=(c == 0))
            ax.title.set_color(colors[g])


# ======================================================================
# 8. Figure
# ======================================================================


def plot_ism_both_directions(
    gene,
    exp_id,
    sum_lfc,
    ref_seq,
    genes_dfs,
    gene_df,
    gff_db,
    meme_path,
    deep_plant_path=None,
    show_tracks=True,
    show_alt=True,
    exp_label=None,
    chip_filter=True,
    pad=50,
    seq_len=2500,
    max_dist=5,
    flank=5000,
    n_logo_cols=6,
    figsize=None,
    panel_letters=True,
):
    """One gene, one experiment: activator (blue) and repressor (red) sites.

    sum_lfc   : (4, L) raw ISM (mut - ref) for this experiment; do NOT multiply
                by a regulation sign.
    genes_dfs : {f"{gene}_{exp_id}_{reg}": analyze_mutations_with_chip(...)},
                reg in (1, -1)
    Returns (fig, regulator_table); fig is None when nothing passes the filters.
    """
    table = regulator_table(genes_dfs, gene, exp_id, chip_filter, max_dist)
    if table.empty:
        print("No paired, ChIP-supported loss-of-function sites in either direction.")
        return None, table

    def positions(reg):
        sel = table.direction == REG_LABELS[reg]
        return set(table.loc[sel, "sequence_relative_position"].astype(int))

    up_pos, down_pos = positions(1), positions(-1)
    all_mut = up_pos | down_pos
    start, end = max(0, min(all_mut) - pad), min(seq_len, max(all_mut) + pad)
    region_length, center = end - start, seq_len // 2

    grow = gene_df.loc[gene_df.gene == gene].iloc[0]
    tss, strand, chrom = int(grow.tss), str(grow.strand), str(grow.chrom)

    groups = _regulator_pwm_groups(table)
    n_logo_rows = sum(math.ceil(len(v) / n_logo_cols) for v in groups.values())
    n_ism = 2 if show_alt else 1
    has_gt = show_tracks and deep_plant_path is not None

    # ---------------------------------------------------------------- layout
    heights = ([3] if has_gt else []) + [1.0 * n_ism + 0.3, 0.55 * max(n_logo_rows, 1)]
    fig = plt.figure(figsize=figsize or (7.2, sum(heights) + 0.4), dpi=300)
    fig.patch.set_facecolor("white")
    outer = fig.add_gridspec(len(heights), 1, height_ratios=heights, hspace=0.15)

    if has_gt:
        pdf = _make_gene_genometracks_pdf(
            deep_plant_path, gene, exp_id, chrom, tss, strand, start, end, center, flank
        )
        ax_track = fig.add_subplot(outer[0])
        ax_track.axis("off")
        ax_track.imshow(_trim_whitespace(convert_from_path(pdf, dpi=300)[0]))
        ax_track.set_aspect("auto")
    ism_spec, logo_spec = outer[int(has_gt)], outer[int(has_gt) + 1]
    gs_ism = GridSpecFromSubplotSpec(
        n_ism + 1,
        1,
        subplot_spec=ism_spec,
        hspace=0.12,
        height_ratios=[1] * n_ism + [0.5],
    )

    # ---------------------------------------------------------------- ISM tracks
    ax_ref = fig.add_subplot(gs_ism[0])
    ks, x_left = _draw_ref_track(
        ax_ref, sum_lfc, ref_seq, all_mut, up_pos, down_pos, start, end, region_length
    )
    ism_axes = [ax_ref]
    if show_alt:
        ax_alt = fig.add_subplot(gs_ism[1])
        _draw_alt_track(
            ax_alt,
            sum_lfc,
            ref_seq,
            all_mut,
            _alt_bases(table),
            start,
            end,
            region_length,
        )
        ism_axes.append(ax_alt)

    lo = min(a.get_ylim()[0] for a in ism_axes)
    hi = max(a.get_ylim()[1] for a in ism_axes)
    for a in ism_axes:
        a.set_ylim(lo, hi)
        _add_y_scale(a)

    _draw_annotation_map(
        fig.add_subplot(gs_ism[n_ism]),
        gene,
        start,
        end,
        x_left,
        ks,
        gff_db,
        TSS_CENTER=center,
        SEQ_LEN=seq_len,
    )

    # ---------------------------------------------------------------- labels
    txt = dict(
        ha="left", va="top", fontsize=FS["tiny"], fontweight="bold", color="#333333"
    )
    ax_ref.text(
        0.005,
        0.97,
        f"Reference allele · {exp_label or f'exp {exp_id}'}",
        transform=ax_ref.transAxes,
        **txt,
    )
    ax_ref.legend(
        handles=[
            Patch(
                facecolor=UP_COLOR,
                alpha=0.35,
                label="Activator site (loss ↓ expression)",
            ),
            Patch(
                facecolor=DOWN_COLOR,
                alpha=0.35,
                label="Repressor site (loss ↑ expression)",
            ),
        ],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=2,
        frameon=False,
        fontsize=FS["tiny"],
        handlelength=1.0,
        handleheight=0.8,
        borderaxespad=0.1,
        columnspacing=1.0,
    )
    if show_alt:
        g0 = relative_to_genomic_position(start, tss, strand, center)
        g1 = relative_to_genomic_position(end - 1, tss, strand, center)
        ax_alt.text(
            0.005, 0.97, "Alternative allele", transform=ax_alt.transAxes, **txt
        )
        ax_alt.text(
            0.995,
            0.97,
            f"{gene} · Chr{chrom}:{min(g0, g1)}–{max(g0, g1)} · strand {strand}",
            transform=ax_alt.transAxes,
            ha="right",
            va="top",
            fontsize=FS["tiny"],
            color="#333333",
        )

    # ---------------------------------------------------------------- logos
    _plot_regulator_logos(fig, logo_spec, meme_path, groups, n_cols=n_logo_cols)

    # ---------------------------------------------------------------- align / letters
    if has_gt:
        p, tp = ax_ref.get_position(), ax_track.get_position()
        ax_track.set_position([p.x0, tp.y0, p.width, tp.height])
    if panel_letters:
        specs = ([outer[0]] if has_gt else []) + [ism_spec, logo_spec]
        for letter, spec in zip("ABC", specs):
            _panel_letter(fig, spec, letter, specific_dx=-6)
    return fig, table
