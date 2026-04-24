import os
import argparse
import re
import torch
import pandas as pd
import h5py
from dotenv import load_dotenv
from numpy import expm1

# Import DeepPlant specific modules
from src.DeepPlant import build_model
from src.config import CSPConfig, GEPConfig, EAPConfig
from src.utils import (
    hot_encode_sequence,
    read_json,
    get_device,
    get_DNA_sequence,
    read_fasta_file,
    create_path,
)


def parse_locus(locus_str):
    """Parses format 'Chr1:1000-3500' into (1, 1000, 3500)"""
    try:
        match = re.match(r"Chr(\d+):(\d+)-(\d+)", locus_str, re.IGNORECASE)
        if not match:
            raise ValueError("Locus must be in format 'Chr1:1000-3500'")
        chrom = int(match.group(1))
        start = int(match.group(2))
        end = int(match.group(3))
        return chrom, start, end
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid locus format: {e}")


def main():
    parser = argparse.ArgumentParser(description="DeepPlant Prediction Script")

    # Required Arguments
    parser.add_argument("--species", choices=["arabidopsis", "rice"], required=True)
    parser.add_argument("--task", choices=["CSP", "GEP", "EAP"], required=True)
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output folder"
    )

    # Mutually Exclusive Group (Fasta OR Locus OR Gene)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fasta", type=str, help="Path to fasta file")
    group.add_argument(
        "--locus", type=parse_locus, help="Locus string e.g. Chr1:1000-3500"
    )
    group.add_argument("--gene", type=str, help="Gene name e.g. AT5G52310")

    args = parser.parse_args()
    if (args.species == "rice") and (args.task == "EAP"):
        raise ValueError(
            "Unfortunately DeepPlant was not trained to predict enhancers in rice."
        )
    # 1. Environment Setup
    load_dotenv()
    deep_plant_path = os.getenv("DEEPPLANTPATH")
    if not deep_plant_path:
        raise ValueError("DEEPPLANTPATH missing from .env file.")

    device = get_device()
    abbr = {"arabidopsis": "AT", "rice": "OS"}.get(args.species)
    species_folder = "arabidopsis" if abbr == "AT" else "oryza"
    # 2. Sequence Retrieval
    if args.fasta:
        # Assuming read_fasta_file returns a single sequence string
        for record in read_fasta_file(args.fasta):
            DNA_sequence = record.seq
            seq_name = record.description
    elif args.gene:
        h5_path = os.path.join(
            deep_plant_path, f"data/{species_folder}/GEP/expression_data_TSS.h5"
        )
        h5_gex = h5py.File(h5_path)
        gene_idx = None
        for i, record in enumerate(h5_gex["records"]):
            if args.gene.upper() in record.decode():
                gene_idx = i
                DNA_sequence = h5_gex["sequences"][gene_idx].decode()
                seq_name = record.decode()
                break
        if not gene_idx:
            raise ValueError(
                f"Gene {args.gene} not found in the gene list for {args.species}."
            )

    else:
        chrom, start, end = args.locus
        DNA_sequence = get_DNA_sequence(
            organism=args.species.capitalize(), chrom=chrom, start=start, end=end
        )
        seq_name = f"{args.species}_Chr{chrom}_{start}_{end}"

    # 3. Model & Path Mapping
    # Logic mapping species shorthand to config/model naming conventions

    config_file = os.path.join(
        deep_plant_path, f"config/config_{abbr}_{args.task}.json"
    )

    if args.task != "EAP":
        file = (
            f"{abbr}_{args.task}_Metadata.tsv"
            if args.task == "GEP"
            else f"{abbr}_{args.task}_Metadata.csv"
        )
        sep = "\t" if args.task == "GEP" else ","
        metadata_file = os.path.join(
            deep_plant_path, f"data/{species_folder}/{args.task}/{file}"
        )
        if os.path.exists(metadata_file):
            metadata = pd.read_csv(metadata_file, sep=sep)
            if args.task == "GEP":
                metadata.rename(columns={"Sample": "Target"}, inplace=True)
                metadata = metadata[["Target", "Tissue", "Genotype", "Treatment"]]
            else:
                metadata = metadata[
                    [
                        "Target",
                        "Factor",
                        "Factor_type_specific",
                        "Tissue",
                        "Mutant_specific",
                        "Age",
                        "Temperature",
                    ]
                ]
    model_file = os.path.join(deep_plant_path, f"models/model_{abbr}_{args.task}.pt")
    if (args.task == "GEP") or (args.task == "CSP"):
        label_file = os.path.join(
            deep_plant_path, f"data/{species_folder}/{args.task}/{args.task}_label.txt"
        )

    configcalss = {"CSP": CSPConfig, "GEP": GEPConfig, "EAP": EAPConfig}.get(args.task)
    config = configcalss(**read_json(config_file))
    if args.task == "CSP":
        config.consistency_regularization = False

    # 4. Load Model
    model = build_model(
        args=config,
        new_model=False,
        task=args.task,
        model_path=model_file,
    ).to(device)
    model.eval()

    # 5. Inference
    with torch.no_grad():
        encoded_seq = hot_encode_sequence(DNA_sequence, length_after_padding=2500)
        input_tensor = torch.from_numpy(encoded_seq).unsqueeze(0).to(device)
        outputs = model(input_tensor)
        if args.task == "EAP":
            activation_fn = torch.nn.Softmax(dim=0)
            outputs = activation_fn(outputs)
        if device != "cpu":
            outputs = outputs.detach().cpu().numpy()
        else:
            outputs = outputs.numpy()
        if args.task == "GEP":
            outputs = expm1(outputs)

    create_path(args.output)
    output_path = os.path.join(args.output, f"{seq_name}_{args.task}_results.csv")
    # 6. Save Results
    if args.task != "EAP":
        targets = open(label_file).read().splitlines()
        pd.DataFrame({"Target": targets, "Prediction": outputs}).merge(
            metadata, how="left", on="Target"
        ).to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
    else:
        print(
            f"The probability of the presence of an enhancer in the center of the given DNA sequence is {outputs[1] * 100:.3f}%"
        )

    print(f"--- Process Complete ---")
    print(f"Species: {args.species} | Task: {args.task}")


if __name__ == "__main__":
    main()
