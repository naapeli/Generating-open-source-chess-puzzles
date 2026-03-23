#!/bin/bash -l
#SBATCH --time=2-00:00:00
#SBATCH --output=outputrl.out
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100:1
# #SBATCH --partition=gpu-b300-288g-short
# #SBATCH --gpus=1
# #SBATCH --partition=gpu-debug
#SBATCH --cpus-per-task=16


module load mamba
module load triton/2024.1-gcc gcc/12.3.0  # needed for torch.compile
source activate environment
srun python src/train_rl.py   # --continue_from_checkpoint # --distributed
