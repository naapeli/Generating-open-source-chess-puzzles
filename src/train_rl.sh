#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --output=outputrl.out
#SBATCH --mem=128G
#SBATCH --nodes=1
# #SBATCH --ntasks=8
# #SBATCH --gpus=h200:8
#SBATCH --ntasks=4
#SBATCH --gpus=h200:4
# #SBATCH --gres=min-vram:32g
#SBATCH --cpus-per-task=16


module load mamba
module load triton/2024.1-gcc gcc/12.3.0  # needed for torch.compile
source activate environment
srun python src/train_rl.py --distributed
