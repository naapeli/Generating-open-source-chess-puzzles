#!/bin/bash -l
#SBATCH --job-name=MaskedDiffusion
#SBATCH --time=24:00:00
#SBATCH --output=output.out
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=h100:4
#SBATCH --cpus-per-gpu=10
#SBATCH --mail-user=aatu.selkee@aalto.fi


module load mamba
source activate environment
srun python src/supervised_training.py --distributed
# srun python src/supervised_training.py
