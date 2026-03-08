import sys
import argparse

sys.path.append("/s/chromatin/m/nobackup/ahmed/DeepPlant")
from src.utils import read_fasta_file


def parse_arguments(parser):
    parser.add_argument("-i", "--input", type=str, help="path to the fasta file")
    parser.add_argument("-o", "--output", type=str, help="path to the h5 file")
    parser.add_argument("-l", "--length", type=int, help="length of subsequence")
    parser.add_argument("-s", "--slide", type=int, help="sliding")
    args = parser.parse_args()
    return args


# generate many data files in a directory
def main():
    parser = argparse.ArgumentParser(
        description="Split chromosomes into sub sequence and save in h5 file"
    )
    args = parse_arguments(parser)
    fasta_path = args.input
    h5_file = args.output
    length = args.length
    sliding = args.slide
    for record in read_fasta_file(fasta_path):
        record = record.description
        sequence = record.seq
    record_list = list()
    sequences_list = list()
    for i in range(0,len(sequence),sliding):
        



# entry point
if __name__ == "__main__":
    main()
