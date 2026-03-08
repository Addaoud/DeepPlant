import argparse
import pandas as pd

# import openpyxl
import glob
import numpy as np
import os
import pyBigWig
import math

# Create an argument parser
parser = argparse.ArgumentParser(description="Label data")
# Add arguments to the parser
parser.add_argument(
    "--input-file",
    type=str,
    default="genomes_chunked_10kb_non_masked.csv",
    help="Name of the clustered genomes data",
)
parser.add_argument(
    "--input-loc",
    type=str,
    default="Bigwig/",
    help="The location of the experiment files",
)
parser.add_argument(
    "--loc",
    type=str,
    default="Non_Overlap_avg_2500_200_50_non_masked/",
    help="The location of the experiment files",
)
parser.add_argument("--norm", type=str, default="rpgc", help="Normalization type")
parser.add_argument(
    "--species", type=str, default="Arabidopsis thaliana", help="Normalization type"
)
parser.add_argument(
    "--window", type=int, default=2500, help="window size for max value"
)
parser.add_argument("--slide", type=int, default=50, help="sliding value")
parser.add_argument(
    "--width", type=int, default=200, help="width to extract values from centre"
)
# Parse the arguments
args = parser.parse_args()
size = 10000
# Get the input location from the arguments
filename = args.loc + args.input_file
df = pd.read_csv(filename)
df = df[df["genome"] == args.species].reset_index()
genome = np.array(df["genome"])
sequence_idx = np.array(df["seq_idx"])
start_pos = np.array(df["start_pos"])
end_pos = np.array(df["end_pos"])
chromosome = np.array(df["chromosome"])
# dataset=np.array(df["dataset"])
# folder_list={"Train":0,"Test":1,"Valid":2}
exp_files_ara = glob.glob(args.input_loc + "/*/*")
sorted_exp_ara = sorted(exp_files_ara, key=lambda f: os.path.basename(f))
print(len(sorted_exp_ara))
f = open(f"{args.loc}/exp_label_{str((args.species).split()[0])}.txt", "w+")
count11 = 0
for j in sorted_exp_ara:
    # print(j)
    f.write((j.split("/")[-2] + "/" + j.split("/")[-1]).split(".bw")[0] + "\n")
    # f.write("\n")
    count11 += 1
f.close()
print(count11)
for i in range(int(len(chromosome) / 2), int(len(chromosome))):
    print(i)
    species = genome[i]
    # print(species)
    if species == "Arabidopsis thaliana":
        output_location = (
            args.loc
            + "/"
            + args.norm
            + "_masked_slide_"
            + str(args.slide)
            + "_window_"
            + str(args.window)
            + "_width_"
            + str(args.width)
            + "_10kb/"
        )
        exp_files = sorted_exp_ara
        chromosome_idx = "Chr" + chromosome[i].split()[-1]
    final_output_location = output_location + "/all_files/"
    # os.makedirs(output_location + "/Train", exist_ok=True)
    os.makedirs(output_location + "/all_files", exist_ok=True)
    # os.makedirs(output_location + "/Test", exist_ok=True)
    # os.makedirs(output_location + "/Valid", exist_ok=True)
    output_filename = final_output_location + sequence_idx[i]

    if (
        os.path.exists(output_filename + ".npy")
        and np.sum((np.load(output_filename + ".npy", allow_pickle=True))) != 0
    ):
        pass
        # print(f"exists{output_filename}")
    else:
        start = start_pos[i]
        end = end_pos[i]
        result = np.zeros((len(exp_files), int((size - args.window) // args.slide) + 1))
        for j1 in range(len(exp_files)):
            j = exp_files[j1]
            if True:
                # try:
                bw = pyBigWig.open(j)

                result_temp = []
                count_k = 0
                for k in range(start, end - args.window + 1, args.slide):
                    mid = k + int(args.window) / 2

                    start_pos_1 = mid - int(args.width / 2)
                    end_pos_1 = mid + int(args.width / 2)
                    value = bw.stats(
                        chromosome_idx, int(start_pos_1), int(end_pos_1), type="mean"
                    )
                    # for l in range(len(value)):
                    #    print(l,k)
                    try:
                        if len(value) > 0 and isinstance(value[0], (int, float)):
                            result[j1][count_k] = value[0]
                    except:
                        pass
                    count_k += 1
            # except:
            #    print(f"No {j}")
            # result=np.array(result).reshape(-1,int((end-start)//args.slide)+1,args.width)
        np.save(output_filename, result)
        print(output_filename, result.shape, result[0])
