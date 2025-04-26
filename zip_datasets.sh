#!/bin/bash

# Recorre todos los ficheros .inter y .tsv en todos los subdirectorios de data/
find data/ -type f \( -name "*.inter" -o -name "*.tsv" \) | while read file; do
  if [[ ! -f "${file}.gz" ]]; then  # Solo comprime si no existe el .gz
    echo "Comprimiendo: $file"
    gzip -k "$file"  # -k para mantener el original
  else
    echo "Ya existe comprimido: ${file}.gz, saltando..."
  fi
done