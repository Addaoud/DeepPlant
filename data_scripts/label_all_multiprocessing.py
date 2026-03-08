import os
import pyBigWig
import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import concurrent.futures as cf  # added
import os
def get_bigwig_files(bigwig_dir):
    bigwig_files = []
    experiment_names = []
    for root, _, files in os.walk(bigwig_dir):
        for file in sorted(files):
            if file.endswith(".bw"):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, bigwig_dir)
                bigwig_files.append(full_path)
                experiment_names.append(relative_path.replace(".bw", ""))
    return bigwig_files, experiment_names

# worker: compute mean for one BigWig (keeps exact behavior)
def _mean_signal_for_file(args):
    bigwig_file, chrom, start, end = args
    bw = pyBigWig.open(bigwig_file)
    try:
        m = bw.stats(chrom, start, end, type="mean")[0]
        return 0.0 if m is None else float(m)
    finally:
        bw.close()

def process_bigwig_files(
    fasta_file,
    bigwig_dir,
    output_dir,
    window_size=2500,
    step_size=200,
    center_size=200,
    n_workers=os.cpu_count() - 1  # processes used INSIDE a chromosome
):
    bigwig_files, experiment_names = get_bigwig_files(bigwig_dir)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "experiment_names.npy"), np.array(experiment_names))
    print(f"Saved experiment names in {output_dir}/experiment_names.npy")

    for record in SeqIO.parse(fasta_file, "fasta"):
        chrom = record.id
        print(f"chrom:{chrom}")
        if chrom in ["Chr1","Chr2","Chr3"]:#,"Chr5","Chr6","Chr7","Chr8","Chr9","Chr10"
            continue  # go to the next record without processing
        if chrom not in ["Chr1","Chr2", "Chr3", "Chr4","Chr5","Chr6","Chr7","Chr8","Chr9","Chr10"]:  #["Chr1", "Chr2", "Chr3", "Chr4", 
            break  # keep EXACT behavior

        seq_length = len(record.seq)
        pos = 0
        chr_output_dir = os.path.join(output_dir, chrom)
        fasta_output_path = os.path.join(chr_output_dir, "sequences.fasta")
        
        os.makedirs(chr_output_dir, exist_ok=True)
        fasta_records = []

        # >>> multiprocessing ONLY within this chromosome <<<
        with cf.ProcessPoolExecutor(max_workers=n_workers) as pool:
            while pos + window_size <= seq_length:
                center_start = pos + (window_size // 2) - (center_size // 2)
                center_end = center_start + center_size
                window_seq = record.seq[pos:pos + window_size]
                
                
                chrom = chrom.strip(",")
                if chrom.lower().startswith("chr"):
                        chrom = chrom[3:]
                
                file_name = f"chr_{chrom}_{pos}_{pos+window_size}"
                npy_output_path = os.path.join(chr_output_dir, f"{file_name}.npy")
                if os.path.exists(npy_output_path):
                    pos += step_size
                    continue
                if 'Z' not in window_seq:
                    values_matrix = np.zeros((len(bigwig_files),), dtype=float)

                    # keep your exact chrom normalization/mutation pattern
                    

                    # parallel over BigWigs for THIS window (inside this chromosome)
                    args_iter = [(bw_path, chrom, center_start, center_end) for bw_path in bigwig_files]
                    print(f"chrom:{chrom},,, center_start:{center_start},,,,center_end{center_end}")
                    results = list(pool.map(_mean_signal_for_file, args_iter))
                    values_matrix[:] = results

                    
                    # print(f"npy_output_path{npy_output_path}")
                    np.save(npy_output_path, values_matrix)
                    fasta_records.append(SeqRecord(Seq(str(window_seq)), id=file_name, description=""))

                pos += step_size  # slide

        # no cross-chromosome overlap: we leave the pool context before next chromosome
        SeqIO.write(fasta_records, fasta_output_path, "fasta")

    print(f"Finished processing. Data saved in {output_dir}")


fasta_file="/s/chromatin/a/nobackup/taib/deepPlant/zea_data1/Zea_mays_genome_Chr_style.fasta"
bigwig_dir="/s/chromatin/a/nobackup/taib/deepPlant/Data1/zea_mays/avg_rep_final"
output_dir = "/s/chromatin/a/nobackup/taib/deepPlant/label_chrom/processed_data_slide_200/"

process_bigwig_files(fasta_file, bigwig_dir, output_dir)
