#!/bin/bash

echo "********************************************"
echo "Running hyperparameter tuning for ItemKNN on bgg_24 dataset using Hyperopt..."

python knn_hyper.py --model='ItemKNN' --dataset='bgg_24' --tool=Hyperopt
