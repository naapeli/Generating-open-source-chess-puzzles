#!/bin/bash -l
#SBATCH --time=00:10:00
#SBATCH --gpus=1
#SBATCH --output=output.out
#SBATCH --mem=32000

module load mamba
source activate environment
srun python src/supervised_training.py
