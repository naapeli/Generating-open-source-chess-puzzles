#!/bin/bash -l
#SBATCH --time=12:00:00
#SBATCH --output=nonpuzzle_dataset.out
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64


module load mamba
source activate environment
srun python src/nonpuzzle_dataset.py
