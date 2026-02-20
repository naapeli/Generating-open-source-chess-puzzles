#!/bin/bash -l
#SBATCH --time=12:00:00
#SBATCH --output=output.out
#SBATCH --mem=128G
#SBATCH --nodes=1
# #SBATCH --ntasks=8
# #SBATCH --gpus=h200:8
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --gres=min-vram:32g
#SBATCH --cpus-per-task=16


module load mamba
source activate environment
srun python src/train_rl.py --distributed
