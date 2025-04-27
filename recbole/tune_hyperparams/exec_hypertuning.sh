#!/bin/bash

echo "********************************************"
echo "Running Hyperparameter Tuning for BGG25 Dataset Metadata Continuous with FM"

python FM_metadata_cont_hyper.py --model='FM' --dataset='bgg25_continuous_metadata' --tool=Hyperopt

echo "Finished Hyperparameter Tuning for BGG25 Dataset Metadata Continuous with FM"
echo "********************************************"
echo "Running Hyperparameter Tuning for BGG25 Dataset Metadata Discrete with FM"

python FM_metadata_disc_hyper.py --model='FM' --dataset='bgg25_discrete_metadata' --tool=Hyperopt

echo "Finished Hyperparameter Tuning for BGG25 Dataset Metadata Discrete with FM"
echo "********************************************"
echo "Running Hyperparameter Tuning for BGG25 Dataset Reviews Continuous with FM"

python FM_reviews_cont_hyper.py --model='FM' --dataset='bgg25_continuous_reviews' --tool=Hyperopt

echo "Finished Hyperparameter Tuning for BGG25 Dataset Reviews Continuous with FM"
echo "********************************************"
echo "Running Hyperparameter Tuning for BGG25 Dataset Reviews Discrete with FM"

python FM_reviews_disc_hyper.py --model='FM' --dataset='bgg25_discrete_reviews' --tool=Hyperopt

echo "Finished Hyperparameter Tuning for BGG25 Dataset Reviews Discrete with FM"
echo "********************************************"