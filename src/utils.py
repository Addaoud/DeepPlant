from typing import Dict, Any, Optional, Tuple, List
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from torch import cuda
import json
from Bio import SeqIO
import seaborn as sns

sns.set_theme()


def read_fasta_file(fasta_path: str, format: str = "fasta"):
    for record in SeqIO.parse(fasta_path, format):
        yield record.upper()


def create_path(path: str) -> None:
    """
    Creates path if it does not exists
    """
    os.makedirs(name=path, exist_ok=True)


def read_json(json_path: str):
    with open(json_path) as f:
        data = json.load(f)
    return data


def get_device():
    device = "cuda" if cuda.is_available() else "cpu"
    return device


def generate_UDir(path: str, UID_length: Optional[int] = 6) -> str:
    """
    Generates a UID of length UID_length that shouldn't exist in the provided path.
    """
    UID = "".join([str(random.randint(0, 9)) for i in range(UID_length)])
    while os.path.exists(os.path.join(path, UID)):
        UID = "".join([str(random.randint(0, 9)) for i in range(UID_length)])
    return UID


def save_data_to_csv(data_dictionary: Dict[str, Any], csv_file_path: str) -> None:
    """
    Save data_dictionary as a new line in a csv file @ csv_file_path
    """
    header = data_dictionary.keys()
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, "w") as fd:
            writer = csv.writer(fd)
            writer.writerow(header)
            writer = csv.DictWriter(fd, fieldnames=header)
            writer.writerow(data_dictionary)
    else:
        with open(csv_file_path, "a", newline="") as fd:
            writer = csv.DictWriter(fd, fieldnames=header)
            writer.writerow(data_dictionary)


def read_excel_csv_file(file_path: str) -> pd.DataFrame:
    """
    Read and return the dataframe at file_path.
    """
    try:
        dataframe = pd.read_csv(file_path)
    except:
        dataframe = pd.read_excel(file_path)
    return dataframe


def save_model_log(log_dir: str, data_dictionary: Dict[str, Any]) -> None:
    """
    save the model logs in the log file
    """
    log_file_path = os.path.join(log_dir, "log_file")
    with open(log_file_path, "a") as log_file:
        if len(data_dictionary) > 0:
            for key, value in data_dictionary.items():
                print("{0}: {1}".format(key, value), file=log_file)
        else:
            print("\n", file=log_file)
            print("".join(["#" for i in range(50)]), file=log_file)
            print("\n", file=log_file)


def plot_loss(loss_csv_path: str, loss_path: str) -> None:
    """
    save the model loss to loss_path
    """
    table = read_excel_csv_file(file_path=loss_csv_path)
    mask = np.isfinite(table.valid_loss.values).tolist()
    plt.xlabel("epoch")
    plt.ylabel("loss value per epoch")
    plt.plot(
        table.epoch,
        table.train_loss,
        "b",
        linestyle="-",
        marker=".",
        label="train_loss",
    )
    plt.plot(
        table.epoch.values[mask],
        table.valid_loss.values[mask],
        "r",
        linestyle="-",
        marker=".",
        label="valid_loss",
    )
    plt.legend()
    plt.savefig(loss_path)
    plt.close()


def count_ambig_bps_in_sequence(DNA_sequence: str) -> int:
    """
    Returns the count of ambiguous base pairs (N,R,Y,W...) in a DNA sequence
    """
    count = 0
    unambig_bps = {"A", "C", "G", "T"}
    for nucl in DNA_sequence.upper():
        if nucl not in unambig_bps:
            count += 1
    return count


def get_paths(start_time: str, model_folder_path: str) -> tuple[str]:
    model_path = os.path.join(
        model_folder_path,
        "model_{0}.pt".format(start_time.strftime("%y_%m_%d:%H:%M")),
    )
    checkpoints_path = os.path.join(
        model_folder_path,
        "checkpoints",
    )
    create_path(checkpoints_path)
    loss_csv_path = os.path.join(
        model_folder_path,
        "loss_{0}.csv".format(start_time.strftime("%y_%m_%d:%H:%M")),
    )
    loss_plot_path = os.path.join(
        model_folder_path,
        "loss_plot_{0}.png".format(start_time.strftime("%y_%m_%d:%H:%M")),
    )
    return model_path, checkpoints_path, loss_csv_path, loss_plot_path


def hot_encode_sequence(
    sequence: str,
    length_after_padding: Optional[int] = 0,
) -> np.ndarray:
    if not hasattr(hot_encode_sequence, "nucleotide_dict"):
        hot_encode_sequence.nucleotide_dict = {
            "A": [1, 0, 0, 0],
            "C": [0, 1, 0, 0],
            "G": [0, 0, 1, 0],
            "T": [0, 0, 0, 1],
            "U": [0, 0, 0, 1],
            "Y": [0, 0, 1 / 2, 1 / 2],
            "R": [1 / 2, 1 / 2, 0, 0],
            "W": [1 / 2, 0, 0, 1 / 2],
            "S": [0, 1 / 2, 1 / 2, 0],
            "K": [0, 1 / 2, 0, 1 / 2],
            "M": [1 / 2, 0, 1 / 2, 0],
            "D": [1 / 3, 1 / 3, 0, 1 / 3],
            "V": [1 / 3, 1 / 3, 1 / 3, 0],
            "H": [1 / 3, 0, 1 / 3, 1 / 3],
            "B": [0, 1 / 3, 1 / 3, 1 / 3],
            "N": [1 / 4, 1 / 4, 1 / 4, 1 / 4],
        }

    if length_after_padding == 0 or length_after_padding < len(sequence):
        hot_encoded_seq = np.zeros((4, len(sequence)), dtype=np.float32)
        start_pos = 0
    else:
        hot_encoded_seq = np.zeros((4, length_after_padding), dtype=np.float32)
        start_pos = (length_after_padding - len(sequence)) // 2

    hot_encoded_seq[:, start_pos : start_pos + len(sequence)] = np.array(
        [
            hot_encode_sequence.nucleotide_dict.get(base, [0, 0, 0, 0])
            for base in sequence
        ],
        dtype=np.float32,
    ).T
    return hot_encoded_seq


def plot_motif_heat(param_matrix):
    param_range = abs(param_matrix).max()
    sns.set_theme(font_scale=2)
    plt.figure(figsize=(param_matrix.shape[1], 4))
    sns.heatmap(
        param_matrix, cmap="PRGn", linewidths=0.2, vmin=-param_range, vmax=param_range
    )
    ax = plt.gca()
    ax.set_xticklabels(range(1, param_matrix.shape[1] + 1))
    ax.set_yticklabels("ACGT", rotation="horizontal")


def parse_arguments(parser):
    parser.add_argument("--json", type=str, help="path to the json file")
    parser.add_argument(
        "-n",
        "--new",
        action="store_true",
        help="Build a new model",
    )
    parser.add_argument("-m", "--model", type=str, help="Existing model path")
    parser.add_argument(
        "-t",
        "--train",
        action="store_true",
        help="Train the model",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        action="store_true",
        help="Evaluate the model",
    )
    args = parser.parse_args()
    assert (
        args.json != None
    ), "Please specify the path to the json file with --json json_path"
    assert os.path.exists(
        args.json
    ), f"The path to the json file {args.json} does not exist. Please verify"
    assert (args.new == True) ^ (
        (args.model) != None
    ), "Wrong arguments. Either include -n to build a new model or specify -m model_path"
    return args
