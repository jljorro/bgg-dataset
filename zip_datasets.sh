#!/bin/bash

find data/ -type f \( -name "*.inter" -o -name "*.tsv" \) | while read file; do
  if [[ ! -f "${file}.gz" ]]; then 
    echo "Compressing: $file"
    gzip -k "$file"  # -k to keep the original
  else
    echo "File already exists: ${file}.gz, ignoring..."
  fi
done