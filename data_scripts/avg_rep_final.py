import pyBigWig
import shutil
import glob
import numpy as np
import os
import subprocess
import argparse
import pandas as pd
from multiprocessing import Pool


def average_bigwig_files_to_bigwig(bigwig_files, output_file):
    F1 = open("error_files.txt", "w+")
    if len(bigwig_files) <= 1:
        shutil.copy(bigwig_files[0], output_file)
        return
    try:
        # if True:
        summary_output = "results.npz"
        cmd_summary = (
            ["multiBigwigSummary", "bins", "-b"] + bigwig_files + ["-o", summary_output]
        )
        subprocess.run(cmd_summary, check=True)
        if len(bigwig_files) == 2:
            cmd_avg = [
                "bigwigCompare",
                "-b1",
                bigwig_files[0],
                "-b2",
                bigwig_files[1],
                "-o",
                output_file,
                "--operation",
                "mean",
            ]
            subprocess.run(cmd_avg, check=True)

            return
        bw_files = [pyBigWig.open(bw) for bw in bigwig_files]
        # Assuming all BigWig files have the same chromosomes and sizes
        chroms = bw_files[0].chroms()

        # Create a new BigWig file for the average
        with pyBigWig.open(output_file, "w") as bw_out:
            bw_out.addHeader(list(chroms.items()))

            for chrom, length in chroms.items():
                # Fetch values from each BigWig file
                values = [bw.values(chrom, 0, length) for bw in bw_files]

                # Compute the average across all replicates
                avg_values = np.nanmean(values, axis=0).tolist()

                # Write the averaged values to the new BigWig file
                bw_out.addEntries(
                    [chrom] * length,
                    list(range(length)),
                    ends=list(range(1, length + 1)),
                    values=avg_values,
                )

        for bw in bw_files:
            bw.close()
    except:
        for item in bigwig_files:
            F1.write("%s\n" % item)


def extract_chrom_sizes_from_bigwig(bigwig_file, output_file):
    """Extract chromosome sizes from a BigWig file."""

    with pyBigWig.open(bigwig_file) as bw:
        with open(output_file, "w") as fout:
            for chrom, length in bw.chroms().items():
                fout.write(f"{chrom}\t{length}\n")


def average_bigwigs_with_deeptools(bigwig_files, output_bigwig):

    # Extract chromosome sizes from the first BigWig file
    chrom_sizes_file = "temp_chrom_sizes.txt"
    extract_chrom_sizes_from_bigwig(bigwig_files[0], chrom_sizes_file)

    # Step 1: Use multiBigwigSummary to compute summary statistics
    summary_output = "results.npz"
    cmd_summary = (
        ["multiBigwigSummary", "bins", "-b"] + bigwig_files + ["-o", summary_output]
    )
    subprocess.run(cmd_summary, check=True)

    # Step 2: Load the summary statistics and compute the average
    data = np.load(summary_output, allow_pickle=True)
    means = np.mean(data["matrix"], axis=1)

    # Step 3: Convert the averaged data to bedGraph format
    bedgraph_output = "averaged.bedGraph"
    with open(bedgraph_output, "w") as fout:
        for start, end, value in zip(
            data["bin_intervals"][:, 0], data["bin_intervals"][:, 1], means
        ):
            fout.write(f"{data['chromosome'][0]}\t{start}\t{end}\t{value}\n")

    # Step 4: Convert bedGraph to BigWig
    cmd_convert = ["bedGraphToBigWig", bedgraph_output, chrom_sizes_file, output_bigwig]
    subprocess.run(cmd_convert, check=True)

    # Clean up temporary files
    os.remove(summary_output)
    os.remove(bedgraph_output)
    os.remove(chrom_sizes_file)


parser = argparse.ArgumentParser(description="Data Download")
parser.add_argument(
    "--species-name",
    type=str,
    default="oryza_sativa",
    metavar="N",
    help="Either oryza_sativa or aradopsis_thaliana",
)
parser.add_argument(
    "--data-path",
    type=str,
    default="/s/chromatin/b/nobackup/deepplant/Data/",
    metavar="N",
    help="Path to downloaded experiment data",
)
parser.add_argument(
    "--met-file",
    type=str,
    default="/s/chromatin/b/nobackup/deepplant/Data/New_Metadata_",
    metavar="N",
    help="Location of metadat file",
)
parser.add_argument(
    "--output-path",
    type=str,
    default="/s/chromatin/b/nobackup/deepplant/Data/oryza_sativa/avg_rep_final/",
    metavar="N",
    help="Path to downloaded experiment data",
)
args = parser.parse_args()

path_to_exp = args.data_path + args.species_name + "/Chiphub_Final/"
output_path = args.data_path + args.species_name + "/avg_final_rpgc/"
df = pd.read_csv(args.met_file + args.species_name + ".csv")
exp_name = np.array(df["Experiment Name"])
projid = np.array(df["BioProject ID"])
proj_exp = np.column_stack((projid, exp_name))
proj_exp = proj_exp.reshape(-1, 2)
uproj_exp = np.array(list(set(tuple(row) for row in proj_exp)))


for i in range(len(uproj_exp)):
    indices = df[df["BioProject ID"] == uproj_exp[i][0]].index.tolist()
    dir1 = args.output_path + uproj_exp[i][0] + "/"
    os.makedirs(dir1, exist_ok=True)
    exp_list = []
    for j in indices:
        if df["Experiment Name"][j] not in exp_list:
            exp_list.append(df["Experiment Name"][j])
    list1 = [[] for k in range(len(exp_list))]
    exp_list = np.array(exp_list)
    for j in indices:
        if df["Sample name"][j] != "control":
            final_file = (
                path_to_exp
                + projid[j]
                + "/signal/"
                + df["Sample accession"][j]
                + ".final.rpgc.bw"
            )
            # print(final_file)
            if os.path.exists(final_file):
                # pass
                list1[int(np.where(df["Experiment Name"][j] == exp_list)[0][0])].append(
                    final_file
                )
                # print(projid[j],df["Experiment Name"][j],df["Replicate"][j],df["Sample name"][j],df["Sample accession"][j])
            else:
                pass
                # print(f"No {final_file}")
    for k in range(len(list1)):
        if len(list1[k]) > 0:

            output_file_name = dir1 + exp_list[k] + ".bw"
            if os.path.exists(output_file_name):
                print(f"exists {output_file_name}")
            else:
                if len(list1[k]) > 1:
                    average_bigwig_files_to_bigwig(list1[k], output_file_name)
                else:
                    cmd1 = f"cp {list1[k][0]} {output_file_name}"
                    print(f"Inside   {cmd1}")
                    os.system(cmd1)
                print(f"Done {list1[k]},{output_file_name}")
    print("-------------")
