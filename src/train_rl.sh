#!/bin/bash -l
#SBATCH --time=01:00:00
#SBATCH --output=output.out
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --gres=min-vram:32g
#SBATCH --cpus-per-gpu=10


module load mamba
source activate environment
srun python src/train_rl.py
