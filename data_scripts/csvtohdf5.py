import pandas as pd
import h5py
import numpy as np


csv_file = (
    "/s/chromatin/a/nobackup/taib/deepPlant/zea_blast_database/all_results_filtered.csv"
)
train_indices_path = "/s/chromatin/a/nobackup/taib/deepPlant/label_chrom/Train.txt"
valid_indices_path = "/s/chromatin/a/nobackup/taib/deepPlant/label_chrom/Val.txt"
test_indices_path = "/s/chromatin/a/nobackup/taib/deepPlant/label_chrom/Test.txt"

train_h5_file = "/s/chromatin/a/nobackup/DeepPlant/zea_mays/train_seq.h5"
valid_h5_file = "/s/chromatin/a/nobackup/DeepPlant/zea_mays/valid_seq.h5"
test_h5_file = "/s/chromatin/a/nobackup/DeepPlant/zea_mays/test_seq.h5"

train_indices = open(train_indices_path).read().splitlines()
valid_indices = open(valid_indices_path).read().splitlines()
test_indices = open(test_indices_path).read().splitlines()


df = pd.read_csv(csv_file)


for indices, h5_file in zip(
    [train_indices, valid_indices, test_indices],
    [train_h5_file, valid_h5_file, test_h5_file],
):
    print("writing ", h5_file)
    with h5py.File(h5_file, "w") as f:
        df_temp = df.loc[df.record.isin(indices)]
        # Save sequences as fixed-length ASCII strings
        max_len = 2500  # df["sequence"].str.len().max()
        dt = h5py.string_dtype(encoding="ascii", length=max_len)
        f.create_dataset("records", data=df_temp["record"].astype("S"))
        f.create_dataset("sequences", data=df_temp["sequence"].astype(dt))
