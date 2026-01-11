#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH -p mhigh # Partition to submit to
#SBATCH --mem 24G # 24GB memory
#SBATCH --gres gpu:1 # Request of 1 gpu
#SBATCH -o logs/%x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e logs/%x_%u_%j.err # File to which STDERR will be written

python3 -m experiments.experiment_double_pyramidal_default --dropout 0.05 --weight-decay 0.0001 --label-smoothing 0.1 --epochs 500
