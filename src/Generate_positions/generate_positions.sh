#!/bin/bash -l
#SBATCH --time=0-05:00:00
#SBATCH --output=generations.out
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-gpu=10


module load mamba
module load triton/2024.1-gcc gcc/12.3.0  # needed for torch.compile
source activate environment
srun python src/generate_positions.py
