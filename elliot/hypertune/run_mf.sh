#!/bin/bash

SCRIPTS=("MF")

for script in ${SCRIPTS[@]}; do
    echo "Running script ${script}..."
    python3 "run_${script}.py" > logs/$script.log 2>&1 &
    echo "${script} finished!"
done

wait
