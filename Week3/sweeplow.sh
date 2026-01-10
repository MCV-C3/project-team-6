#!/bin/bash
#SBATCH -n 8 # Number of cores
#SBATCH -p mlow # Partition to submit to
#SBATCH --array=1-50
#SBATCH --mem 24G # 24GB memory
#SBATCH --gres gpu:1 # Request of 1 gpu
#SBATCH -o logs/%x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e logs/%x_%u_%j.err # File to which STDERR will be written

wandb agent --count 1 mcv-team-6/C3-Week3/k5ns8jzl
