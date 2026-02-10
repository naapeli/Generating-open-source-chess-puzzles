#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --output=output.out
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=h100:4
#SBATCH --cpus-per-task=32


module load mamba
source activate environment
srun python src/train_rl.py --distributed
