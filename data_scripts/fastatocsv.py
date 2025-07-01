import pandas as pd
import os
import sys

sys.path.append("/s/chromatin/m/nobackup/ahmed/DeepPlant/")
from src.utils import read_fasta_file
import argparse


def parse_arguments(parser):
    parser.add_argument("-i", "--input", type=str, help="path to the input fasta file")
    parser.add_argument(
        "-o", "--output", type=str, help="path to the output fasta file"
    )
    args = parser.parse_args()
    return args


# generate many data files in a directory
def main():
    parser = argparse.ArgumentParser(description="Train and evaluate LR model")
    args = parse_arguments(parser)
    fasta_file_path = args.input
    records = list()
    sequences = list()
    for record in read_fasta_file(fasta_file_path):
        records.append(record.description)
        sequences.append(record.seq)
    pd.DataFrame({"record": records, "sequence": sequences}).to_csv(
        os.path.join(os.path.dirname(fasta_file_path), args.output),
        index=False,
    )


# entry point
if __name__ == "__main__":
    main()
