#!/bin/bash

# Carpeta donde están los scripts
SCRIPTS_DIR="."  # o "." si es el mismo directorio

# Para cada script .py en la carpeta
for script in "$SCRIPTS_DIR"/*.py; do
    (
        script_name=$(basename "$script")
        echo "Lanzando $script_name..."
        python3 "$script"
        echo "$script_name ha terminado."
    ) &
done

# Esperamos a que todos terminen
wait

echo "Todos los scripts han terminado."