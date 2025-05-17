#!/bin/bash

# Match the pattern ALG*_f*/ in every folder
for dir in */*/; do
    # Extract the name of algorithm and fold from folder path
    alg_fold=$(basename "$(dirname "$dir")")_$(basename "$dir")
    alg_fold=${alg_fold%/}

    max_it=-1
    selected_file=""
    fallback_file=""

    for file in "$dir"*.tsv; do
        [[ -f "$file" ]] || continue

        # If it has it=XX, we evaluate it
        if [[ "$file" =~ it=([0-9]+) ]]; then
            it_val=${BASH_REMATCH[1]}
            if (( it_val > max_it )); then
                max_it=$it_val
                selected_file="$file"
            fi
        elif [[ -z "$fallback_file" ]]; then
            # Use the fallback as the first file without it=XX that we found
            fallback_file="$file"
        fi
    done

    # If we find one with it=XX, we use it. Otherwise, we use the fallback if exists.
    if [[ -n "$selected_file" ]]; then
        new_name="${alg_fold}.tsv"
        mv "$selected_file" "./$new_name"
        echo "Moved (with it): $selected_file → $new_name"
    elif [[ -n "$fallback_file" ]]; then
        new_name="${alg_fold}.tsv"
        mv "$fallback_file" "./$new_name"
        echo "Moved (fallback): $fallback_file → $new_name"
    fi
done
