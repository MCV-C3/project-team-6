#!/bin/bash
#SBATCH -n 1
#SBATCH -t 0-00:01
#SBATCH -p mhigh
#SBATCH --mem=1G
#SBATCH -o test_%j.out
#SBATCH -e test_%j.err

echo "Job started on $(hostname)"
echo "PWD is $(pwd)"
which python3
python3 --version
echo "Done"
