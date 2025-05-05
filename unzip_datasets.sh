#!/bin/bash

# Recorre todos los ficheros .gz en data/ y subdirectorios
find data/ -type f -name "*.gz" | while read file; do
  target="${file%.gz}"  # El nombre del fichero descomprimido
  if [[ ! -f "$target" ]]; then  # Solo descomprime si no existe el fichero destino
    echo "Descomprimiendo: $file"
    gunzip -k "$file"  # -k para mantener el .gz original
  else
    echo "Ya existe descomprimido: $target, saltando..."
  fi
done