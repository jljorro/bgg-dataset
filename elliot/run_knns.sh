#!/bin/bash

SCRIPTS=("ItemKNN" "UserKNN")

for script in ${SCRIPTS[@]}; do
    echo "Ejecutndo script ${script}..."
    python3 "run_${script}.py" > logs/$script.log 2>&1 &
    echo "${script} finalizado!"
done

wait
