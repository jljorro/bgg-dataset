#!/bin/bash

SCRIPTS=("ItemKNN" "UserKNN")

for script in ${SCRIPTS[@]}; do
    echo "Ejecutndo script ${script}..."
	python3 "run_${script}.py"
	echo "${script} finalizado!"
done

wait
