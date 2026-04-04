#!/bin/bash

input_base="/s/chromatin/m/nobackup/ahmed/DeepPlant/haoxuan/motifresult"
output_base="/s/chromatin/m/nobackup/ahmed/DeepPlant/haoxuan/stremeresult"

for fasta_file in "$input_base"/filter*/top_motifs_*.fasta; do
    filter_dir=$(basename "$(dirname "$fasta_file")")

    output_dir="$output_base/$filter_dir"

    if [ -f "$output_dir/streme.txt" ]; then
        echo "Skipping $fasta_file, results already exist in $output_dir"
        continue 
    fi
    mkdir -p "$output_dir"

    echo "Running STREME on $fasta_file ..."
    /s/chopin/k/grad/adaoud/meme/bin/streme \
        --verbosity 1 \
        --oc "$output_dir" \
        --dna \
        --totallength 4000000 \
        --time 14400 \
        --minw 8 \
        --maxw 15 \
        --thresh 0.05 \
        --align center \
        --p "$fasta_file"
done

echo "All tasks completed."