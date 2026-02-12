#!/bin/bash -l
#SBATCH --time=01:00:00
#SBATCH --output=output.out
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus=h200:8
#SBATCH --cpus-per-task=16


module load mamba
source activate environment
srun python src/train_rl.py --distributed
