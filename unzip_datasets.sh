#!/bin/bash

find data/ -type f -name "*.gz" | while read file; do
  target="${file%.gz}"
  if [[ ! -f "$target" ]]; then
    echo "Uncompressing: $file"
    gunzip -k "$file"  # -k to keep the original .gz
  else
    echo "File already exists: $target, ignoring..."
  fi
done