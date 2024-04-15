#!/bin/bash
#SBATCH -J nc_metrics 
#SBATCH -o logs/%x.%j.log
#SBATCH --error=logs/%x.%j.err
#SBATCH -p gpuA100x8
#SBATCH --gpus 1
#SBATCH --mem 200G
#SBATCH --ntasks-per-node=1
#SBATCH --account bbjr-delta-gpu
#SBATCH -N 1              # Number of nodes, not cores (16 cores/node)
#SBATCH -n 1             # Total number of MPI tasks 
#SBATCH --time=48:00:00

eval "$(conda shell.bash hook)"
conda activate daniel_jax
echo "daniel_jax activated"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/dbeaglehole/.conda/envs/daniel_jax/lib
echo "Delta setup"

cd /u/dbeaglehole/neural_collapse_rfm

python -u nc_nn.py -dataset mnist -n 50000 -epochs 15000
#python -u nc_nn.py -dataset cifar -n 50000 -epochs 15000
#python -u nc_nn.py -dataset svhn -n 50000 -epochs 15000
