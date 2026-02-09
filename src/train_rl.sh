#!/bin/bash -l
# #SBATCH --time=00:30:00
# #SBATCH --output=output.out
# #SBATCH --mem=64G
# #SBATCH --gpus=1
# #SBATCH --gres=min-vram:32g
# #SBATCH --cpus-per-gpu=10


# module load mamba
# source activate environment
# srun python src/train_rl.py

#SBATCH --time=00:30:00
#SBATCH --output=output.out
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=2
# #SBATCH --gpus=h100:4
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32


module load mamba
source activate environment
srun python src/train_rl.py --distributed
