#!/bin/bash

# Recorremos todos los directorios que coincidan con el patrón ALG*_f*/
for dir in */*/; do
    # Extraemos el nombre del algoritmo y el fold a partir del path del directorio
    alg_fold=$(basename "$(dirname "$dir")")_$(basename "$dir")
    alg_fold=${alg_fold%/}

    max_it=-1
    selected_file=""
    fallback_file=""

    for file in "$dir"*.tsv; do
        [[ -f "$file" ]] || continue

        # Si tiene it=XX, lo evaluamos
        if [[ "$file" =~ it=([0-9]+) ]]; then
            it_val=${BASH_REMATCH[1]}
            if (( it_val > max_it )); then
                max_it=$it_val
                selected_file="$file"
            fi
        elif [[ -z "$fallback_file" ]]; then
            # Usamos como fallback el primer archivo sin it=XX que encontremos
            fallback_file="$file"
        fi
    done

    # Si encontramos uno con it=XX, lo usamos. Si no, usamos el fallback si existe.
    if [[ -n "$selected_file" ]]; then
        new_name="${alg_fold}.tsv"
        mv "$selected_file" "./$new_name"
        echo "Movido (con it): $selected_file → $new_name"
    elif [[ -n "$fallback_file" ]]; then
        new_name="${alg_fold}.tsv"
        mv "$fallback_file" "./$new_name"
        echo "Movido (fallback): $fallback_file → $new_name"
    fi
done
