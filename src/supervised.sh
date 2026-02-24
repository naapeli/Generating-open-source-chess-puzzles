#!/bin/bash -l
#SBATCH --job-name=MaskedDiffusion
# #SBATCH --time=02-00
#SBATCH --time=12:00:00
#SBATCH --output=output.out
#SBATCH --mem=128G
#SBATCH --nodes=1
# #SBATCH --ntasks=8
# #SBATCH --gpus=h200:8
#SBATCH --ntasks=4
#SBATCH --gpus=a100:4
# #SBATCH --ntasks=4
# #SBATCH --gpus=4
# #SBATCH --gres=min-vram:80g
#SBATCH --cpus-per-gpu=10


module load mamba
module load triton/2024.1-gcc gcc/12.3.0  # needed for torch.compile
source activate environment
srun python src/supervised_training.py --distributed
