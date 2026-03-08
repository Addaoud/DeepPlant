import numpy as np
import os


# Function to read a MEME database file and convert LPMs to PFMs
def convert_meme_to_pfm(file_path):
    # Dictionary to store PFMs for each motif
    motif_pfms = {}

    with open(file_path, "r") as file:
        lines = file.readlines()

    # Initialize variables for parsing
    motif_name = None
    nsites = None
    lpm = []
    is_reading_matrix = False

    for line in lines:
        line = line.strip()

        if line.startswith("MOTIF"):

            if motif_name and lpm:
                lpm_array = np.array(lpm)
                pfm = (
                    np.round(lpm_array * nsites).astype(int).T
                )  # Convert to PFM and transpose
                motif_pfms[motif_name] = pfm

            parts = line.split()
            motif_name = parts[1]  # Extract motif name
            nsites = None
            lpm = []
            is_reading_matrix = False

        elif line.startswith("letter-probability matrix"):
            nsites = int(line.split("nsites=")[1].split()[0])
            is_reading_matrix = True

        elif line.startswith("URL"):
            is_reading_matrix = False

        elif is_reading_matrix:
            if line:
                probabilities = list(map(float, line.split()))
                lpm.append(probabilities)
            else:
                is_reading_matrix = False

    if motif_name and lpm:
        lpm_array = np.array(lpm)
        pfm = np.round(lpm_array * nsites).astype(int).T
        motif_pfms[motif_name] = pfm

    return motif_pfms


def write_pfm_to_file(pfm, output_file_path):
    with open(output_file_path, "w") as file:
        for row in pfm:

            file.write(" ".join(map(str, row)) + "\n")


# for file_name in os.listdir(
#     "/s/chromatin/m/nobackup/ahmed/DeepPlant/data/arabidopsis/Jaspar"
# ):
# Input file path
# if file_name == "all.meme":
#     continue
file_name = "all.meme"
# file_path = "/s/chromatin/m/nobackup/ahmed/DeepPlant/data/arabidopsis/Jaspar/all.meme"  # Path to your input file containing the LPM
pfms = convert_meme_to_pfm(
    os.path.join(
        "/s/chromatin/m/nobackup/ahmed/DeepPlant/data/arabidopsis/Jaspar", file_name
    )
)

for motif, pfm in pfms.items():
    print(f"Writing PFM for motif {motif}:")
    write_pfm_to_file(
        pfm,
        f"/s/chromatin/m/nobackup/ahmed/DeepPlant/data/arabidopsis/PFMs/{motif}.pfm",
    )
    print()
