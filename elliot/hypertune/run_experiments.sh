#!/bin/bash

SCRIPTS_DIR="."

for script in "$SCRIPTS_DIR"/*.py; do
    (
        script_name=$(basename "$script")
        echo "Running $script_name..."
        python3 "$script"
        echo "$script_name has finished."
    ) &
done

wait

echo "Every script has finished."