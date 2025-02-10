from src.utils import create_path
import os
import numpy as np
from typing import Optional
from math import sqrt
from math import ceil
from fastprogress import progress_bar
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


def clip(value: float, threshold: Optional[float] = 50):
    return min(value, threshold + sqrt(max(0, value - threshold)))


# save data to a file
def save_file(filepath, data):
    # open the file
    np.save(filepath, data)


def load_file(filepath):
    return np.array(list(map(clip, np.load(filepath))))


def load_save_files(old_files):
    print("running now")
    for old_file in progress_bar(old_files):
        old_filepath = os.path.join(labels_path, old_file)
        new_filepath = os.path.join(new_labels_path, old_file)
        if not os.path.exists(new_filepath):
            cliped_labels = load_file(old_filepath)
            save_file(new_filepath, cliped_labels)


# generate many data files in a directory
def main():

    global labels_path
    labels_path = "/s/chromatin/c/nobackup/deepplant/Data/Arabidopsis_thaliana/Non_Overlap_avg_2500_200_200/Labels_10kb_masked_2500"
    # labels_path = "/s/chromatin/c/nobackup/deepplant/Data/Arabidopsis_thaliana/Non_Overlap_avg_4096_256_64/Labels_10kb_masked_4096"
    # labels_path = "/s/chromatin/c/nobackup/deepplant/Data/Arabidopsis_thaliana/Non_Overlap_avg/Labels_10kb_masked_2500"

    global new_labels_path
    new_labels_path = "/s/chromatin/c/nobackup/deepplant/Data/Arabidopsis_thaliana/Non_Overlap_avg_2500_200_200/Labels_10kb_masked_2500_cliped"
    # new_labels_path = "/s/chromatin/c/nobackup/deepplant/Data/Arabidopsis_thaliana/Non_Overlap_avg/Labels_10kb_masked_2500_cliped"

    # create a local directory to save files
    create_path(new_labels_path)

    for root, dirs, files in os.walk(labels_path):
        n_files = len(files)

        # determine chunksize
        n_workers = multiprocessing.cpu_count()
        print(n_workers)
        chunksize = ceil(n_files / n_workers)

        # create the process pool
        with ProcessPoolExecutor(n_workers) as exe:
            # split the rename operations into chunks
            for i in range(0, n_files, chunksize):
                # select a chunk of filenames
                old_files = files[i : (i + chunksize)]
                # submit file rename tasks
                _ = exe.submit(load_save_files, old_files)
    print("Done\n")


# entry point
if __name__ == "__main__":
    main()
