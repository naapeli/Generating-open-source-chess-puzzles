#!/bin/bash -l
#SBATCH --time=00:10:00
#SBATCH --output=output.out
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=10


module load mamba
source activate environment
srun python src/train_rl.py
