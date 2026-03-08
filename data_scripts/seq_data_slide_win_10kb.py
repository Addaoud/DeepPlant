import argparse
import pandas as pd
import openpyxl
import glob
import numpy as np
import os
import pyBigWig
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO


def read_fasta(fasta_file):
    seq_list = []
    with open(fasta_file, "r") as file:
        sequence = ""
        for line in file:
            if line.startswith(">"):
                header = line  # Reset header for the new sequence
                if len(sequence) > 0:
                    seq_list.append("".join(process_string(sequence.upper())))
                sequence = ""
            else:
                sequence += line  # Add lines of the sequence
    seq_list.append("".join(process_string(sequence.upper())))
    return seq_list


# Create an argument parser


def process_string(s):
    # Remove newline characters
    s = s.replace("\n", "")
    # Convert all characters to upper case
    s = s.upper()
    # Remove all whitespace
    s = "".join(s.split())
    return s


Filename = "/s/chromatin/b/nobackup/deepplant/Data/arabidopsis_thaliana/genome/GCF_000001735.4_TAIR10.1_genomic.fna.masked"
parser = argparse.ArgumentParser(description="Label data")
# Add arguments to the parser
parser.add_argument(
    "--input-file",
    type=str,
    default="genomes_chunked_10kb_masked.csv",
    help="Name of the clustered genomes data",
)
parser.add_argument(
    "--input-loc",
    type=str,
    default="/s/chromatin/b/nobackup/deepplant/Data/",
    help="The location of the experiment files",
)
parser.add_argument("--norm", type=str, default="rpgc", help="Normalization type")
parser.add_argument(
    "--species", type=str, default="Arabidopsis thaliana", help="Normalization type"
)
parser.add_argument(
    "--output-loc",
    type=str,
    default="/s/chromatin/c/nobackup/deepplant/Data/",  # arabidopsis_thaliana/",
    help="The output location for the label file",
)
parser.add_argument(
    "--window", type=int, default=2500, help="window size for max value"
)
parser.add_argument("--slide", type=int, default=200, help="sliding value")
parser.add_argument(
    "--width", type=int, default=200, help="width to extract values from centre"
)
parser.add_argument(
    "--padd", type=int, default=200, help="padd to extract values from centre"
)
# Parse the arguments
args = parser.parse_args()
seq = read_fasta(Filename)
print(len(seq))
size = 10000
# Get the input location from the arguments
filename = args.input_loc + args.input_file
df = pd.read_csv(filename)
df = df[df["genome"] == args.species].reset_index()
genome = np.array(df["genome"])
sequence_idx = np.array(df["seq_idx"])
req_temp_no = np.where(sequence_idx == "seq_9542")[0]
start_pos = np.array(df["start_pos"])
end_pos = np.array(df["end_pos"])
chromosome = np.array(df["chromosome"])
# dataset=np.array(df["dataset"])
# folder_list={"Train":0,"Test":1,"Valid":2}
for i in range(len(chromosome)):
    # for i in  req_temp_no:#(len(chromosome)):
    print(i)
    # break
    species = genome[i]
    print(species)
    if species == "Arabidopsis thaliana":
        output_location = (
            args.output_loc
            + "/Arabidopsis_thaliana/"
            + args.norm
            + "_masked_slide_"
            + str(args.slide)
            + "_window_"
            + str(args.window)
            + "_width_"
            + str(args.width)
            + "_max_10kb/"
        )

        chromosome_idx = "Chr" + chromosome[i].split()[-1]
    else:
        output_location = (
            args.output_loc
            + "/oryza_sativa/"
            + args.norm
            + "_label_window_"
            + str(args.window)
            + "_padd_"
            + str(args.padd)
            + "/"
        )
        exp_files = sorted_exp_ory
        chromosome_idx = chromosome[i].split()[-1]
    final_output_location = output_location + "/window_sequence/"
    output_filename = final_output_location + sequence_idx[i]
    if True:
        """if os.path.exists(
            output_filename #+ ".fasta"
        ):  # and np.sum(np.isnan((np.load(output_filename+".npy",allow_pickle=True)).astype("float32")))==0:
            print(f"exists{output_filename}")
        """
        # else:
        start = start_pos[i]
        end = end_pos[i]
        if end - start > 8000:

            result = []
            # result = np.zeros(((int((size - args.window) // args.slide) + 1),args.window))
            count_k = 0
            for k in range(start, end - args.window + 1, args.slide):
                mid = k + int(args.window) / 2

                start_pos_1 = int(k)
                end_pos_1 = int(k + int(args.window))
                # start_pos_1 = int(mid - int(args.width / 2))
                # end_pos_1 = int(mid + int(args.width / 2))
                value = process_string(
                    seq[int(chromosome_idx[-1]) - 1][start_pos_1:end_pos_1]
                )  # bw.values(chromosome_idx, int(start_pos_1), int(end_pos_1))
                # print(f"---->{len(value)},{seq[int(chromosome_idx[-1])-1][start_pos_1:end_pos_1]}")
                result.append(
                    SeqRecord(
                        Seq(value),
                        id=f"Arabidopsis_{chromosome_idx}_{start_pos_1}_{end_pos_1}",
                    )
                )
                # print(result[0][:10])
            # print(f'-->{len(result[0])},{len(np.array(result))}')

            # result=np.array(result)#.reshape((int((size - args.window) // args.slide) + 1),args.window)
            # np.save(output_filename, result)
            print(f"Before {output_filename}")  # ,  result[0])
            SeqIO.write(result, output_filename + ".fasta", "fasta")
            print(f"After {output_filename}")  # ,  result[0])
