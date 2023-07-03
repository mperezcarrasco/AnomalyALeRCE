#!/bin/bash

# Change to the directory containing the script
cd ..
pwd
echo $$

# Run the Python script

python3 main.py --model ae --hierClass Transient --all_outliers True --lr 5e-4 --z_dim 32
python3 main.py --model ae --hierClass Periodic --all_outliers True --lr 1e-3 --z_dim 64
python3 main.py --model ae --hierClass Stochastic --all_outliers True --lr 5e-3 --z_dim 64

python3 main.py --model vae --hierClass Transient --all_outliers True --lr 5e-4 --z_dim 32
python3 main.py --model vae --hierClass Periodic --all_outliers True --lr 1e-3 --z_dim 64
python3 main.py --model vae --hierClass Stochastic --all_outliers True --lr 5e-3 --z_dim 64

python3 main.py --model deepsvdd --hierClass Transient --all_outliers True --lr 5e-4 --z_dim 32
python3 main.py --model deepsvdd --hierClass Periodic --all_outliers True --lr 5e-4 --z_dim 64
python3 main.py --model deepsvdd --hierClass Stochastic --all_outliers True --lr 5e-4 --z_dim 64

python3 main.py --model classvdd --hierClass Transient --all_outliers True --lr 5e-4 --z_dim 32
python3 main.py --model classvdd --hierClass Periodic --all_outliers True --lr 5e-4 --z_dim 64
python3 main.py --model classvdd --hierClass Stochastic --all_outliers True --lr 5e-4 --z_dim 64

python3 main.py --model iforest --hierClass Transient --all_outliers True  
python3 main.py --model iforest --hierClass Periodic --all_outliers True  
python3 main.py --model iforest --hierClass Stochastic --all_outliers True  

python3 main.py --model ocsvm --hierClass Transient --all_outliers True  
python3 main.py --model ocsvm --hierClass Periodic --all_outliers True  
python3 main.py --model ocsvm --hierClass Stochastic --all_outliers True  