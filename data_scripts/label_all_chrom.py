import os
import pyBigWig
import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
def get_bigwig_files(bigwig_dir):
    """
    Recursively finds all .bw files in the given directory.
    
    Args:
        bigwig_dir (str): Path to the BigWig directory.
    
    Returns:
        list: Sorted list of BigWig file paths.
        list: Corresponding experiment names (e.g., Bio1/Exp1, Bio1/Exp2).
    """
    bigwig_files = []
    experiment_names = []

    for root, _, files in os.walk(bigwig_dir):
        for file in sorted(files):
            if file.endswith(".bw"):  # Ensure it's a BigWig file
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, bigwig_dir)
                #if relative_path.replace(".bw", "")!="SRP080945/SRX2000799_Rep0.rpgc": 
                bigwig_files.append(full_path)
                    
                experiment_names.append(relative_path.replace(".bw", ""))

    return bigwig_files, experiment_names

def process_bigwig_files(fasta_file, bigwig_dir, output_dir, window_size=2500, step_size=200, center_size=200):
    """
    Extracts signal values from multiple BigWig files and stores them in NumPy arrays.

    Args:
        fasta_file (str): Path to the input FASTA file.
        bigwig_dir (str): Path to the directory containing BigWig files.
        output_dir (str): Directory to save results.
        window_size (int): Size of the sliding window (default: 2500).
        step_size (int): Step size for the sliding window (default: 200).
        center_size (int): Size of the center region (default: 200).
    """
    bigwig_files, experiment_names = get_bigwig_files(bigwig_dir)
    os.makedirs(output_dir, exist_ok=True)
    # print(f"experiment_names:{experiment_names}")
    np.save(os.path.join(output_dir, "experiment_names.npy"), np.array(experiment_names))
    print(f"Saved experiment names in {output_dir}/experiment_names.npy")

    for record in SeqIO.parse(fasta_file, "fasta"):
        chrom = record.id
        if chrom not in ["Chr1", "Chr2", "Chr3", "Chr4", "Chr5","Chr6","Chr7","Chr8","Chr9","Chr10"]: #
            break
        seq_length = len(record.seq)  
        pos = 0  
        chr_output_dir = os.path.join(output_dir, chrom)
        fasta_output_path = os.path.join(chr_output_dir, "sequences.fasta")
        os.makedirs(chr_output_dir, exist_ok=True)
        fasta_records = []  

        while pos + window_size <= seq_length:
            center_start = pos + (window_size // 2) - (center_size // 2)
            center_end = center_start + center_size
            window_seq = record.seq[pos:pos + window_size]
            if 'Z' not  in window_seq:
                values_matrix = np.zeros((len(bigwig_files),))
                # Process each BigWig file
                for i, bigwig_file in enumerate(bigwig_files):
                    bw = pyBigWig.open(bigwig_file)
                    print(f"chrom:{chrom},,, center_start:{center_start},,,,center_end{center_end}")
                    chrom = chrom.strip(",")   # remove trailing commas
                    if chrom.lower().startswith("chr"):
                          chrom = chrom[3:]
                    mean_signal = bw.stats(chrom, center_start, center_end, type="mean")[0]
                    bw.close()
                    values_matrix[i] = mean_signal if mean_signal is not None else 0
                file_name = f"chr_{chrom}_{pos}_{pos+window_size}"
                npy_output_path = os.path.join(chr_output_dir, f"{file_name}.npy")
                np.save(npy_output_path, values_matrix)
                fasta_records.append(SeqRecord(Seq(str(window_seq)), id=file_name, description=""))
            pos += step_size  # Move sliding window
        SeqIO.write(fasta_records, fasta_output_path, "fasta")

    print(f"Finished processing. Data saved in {output_dir}")



# Example Usage
# fasta_file = "masked_chrom.fasta"  # Update with actual FASTA file path
# bigwig_dir = "/s/chromatin/c/nobackup/deepplant/Data/Arabidopsis_thaliana/Bigwig/"  # Directory containing BigWig files
fasta_file="/s/chromatin/a/nobackup/taib/deepPlant/zea_data1/Zea_mays_genome_Chr_style.fasta"
bigwig_dir="/s/chromatin/a/nobackup/taib/deepPlant/Data1/zea_mays/avg_rep_final"
output_dir = "/s/chromatin/a/nobackup/taib/deepPlant/label_chrom/processed_data_slide_50/"

process_bigwig_files(fasta_file, bigwig_dir, output_dir)
