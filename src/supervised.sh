#!/bin/bash -l
#SBATCH --job-name=MaskedDiffusion
#SBATCH --time=12:00:00
#SBATCH --output=output.out
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=h100:4
#SBATCH --cpus-per-gpu=10


module load mamba
source activate environment
srun python src/supervised_training.py --distributed --continue_from_checkpoint
# srun python src/supervised_training.py
