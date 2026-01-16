#!/bin/bash -l
#SBATCH --time=00:30:00
#SBATCH --output=output.out
#SBATCH --mem=32G
#SBATCH --gpus=1

module load mamba
source activate environment
srun python src/generate_fens.py
