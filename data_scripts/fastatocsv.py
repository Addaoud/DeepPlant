from src.utils import read_fasta_file
import pandas as pd
import os


# generate many data files in a directory
def main():
    fasta_file_path = "/s/chromatin/c/nobackup/deepplant/Data/Arabidopsis_thaliana/Non_Overlap_avg_2500_200_200/2500_Seq_10kb_masked.fasta"
    records = list()
    sequences = list()
    for record in read_fasta_file(fasta_file_path):
        records.append(record.description)
        sequences.append(record.seq)
    pd.DataFrame({"record": records, "sequence": sequences}).to_csv(
        os.path.join(os.path.dirname(fasta_file_path), "2500_Seq_10kb_masked.csv"),
        index=False,
    )


# entry point
if __name__ == "__main__":
    main()
