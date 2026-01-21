#!/bin/bash -l
#SBATCH --time=00:02:00
#SBATCH --output=output.out
#SBATCH --mem=16G
#SBATCH --cpus-per-task=20

module load mamba
source activate environment
srun python src/metrics/themes.py
