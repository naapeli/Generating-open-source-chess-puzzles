#!/bin/bash -l
#SBATCH --time=01:30:00
#SBATCH --output=output.out
#SBATCH --mem=32G
#SBATCH --cpus-per-task=40

module load mamba
source activate environment
# srun python src/label_generations.py
srun python src/get_lichess_metrics.py
