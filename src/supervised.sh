#!/bin/bash -l
#SBATCH --job-name=MaskedDiffusion
#SBATCH --time=0-24:00:00
#SBATCH --output=output.out
#SBATCH --mem=128G
#SBATCH --nodes=1
# #SBATCH --ntasks=8
# #SBATCH --gpus=h200:8
#SBATCH --ntasks=4
#SBATCH --gpus=h200:4
# #SBATCH --constraint="a100|h100|h200"
# #SBATCH --gpus=4
# #SBATCH --ntasks=8
# #SBATCH --gpus=v100:8
#SBATCH --cpus-per-gpu=10


module load mamba
module load triton/2024.1-gcc gcc/12.3.0  # needed for torch.compile
source activate environment
srun python src/supervised_training.py --distributed --continue_from_checkpoint
