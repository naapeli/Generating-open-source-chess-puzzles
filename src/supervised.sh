#!/bin/bash -l
#SBATCH --time=00:10:00
#SBATCH --output=output.out
#SBATCH --mem=32G
# #SBATCH --mem-per-gpu=32G
# #SBATCH --ntasks=2
# #SBATCH --gpus=2
# #SBATCH --cpus-per-gpu=10


module load mamba
source activate environment
# srun python src/supervised_training.py --distributed
srun python src/supervised_training.py
