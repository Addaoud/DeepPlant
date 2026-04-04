#!/bin/bash

base_dir="/s/chromatin/m/nobackup/ahmed/DeepPlant/haoxuan"
streme_dir="${base_dir}/stremeresult"
tomtom_dir="${base_dir}/tomtomresult"
database="/s/chromatin/m/nobackup/ahmed/DeepPlant/haoxuan/ArabidopsisDAPv1.meme"  

for filter_path in "${streme_dir}"/filter*; do
    filter_name=$(basename "$filter_path")
    streme_txt="${filter_path}/streme.txt"

    output_dir="${tomtom_dir}/${filter_name}"
    mkdir -p "$output_dir"

    meme_file="${output_dir}/query.meme"
    awk '
        BEGIN { print_header=1; motif_count=0 }
        /^MOTIF/ { 
            if (motif_count == 1) { exit } 
            motif_count++ 
        }
        {
            if (print_header) {
                print
                if (/^MOTIF/) { print_header=0 }
            } else if (motif_count <= 1) {
                print
            }
        }
    ' "$streme_txt" > "$meme_file"

    if grep -q "^MOTIF" "$meme_file"; then
        echo "Running tomtom for $filter_name..."
        /s/chopin/k/grad/adaoud/meme/bin/tomtom -oc "$output_dir" -no-ssc -verbosity 1 -min-overlap 5 -mi 1 -dist pearson -evalue -thresh 10.0 -time 300 "$meme_file" "$database"
    else
        echo "Warning: No motif found in $streme_txt, skipping $filter_name"
    fi
done
