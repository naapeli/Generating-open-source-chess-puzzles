#!/bin/bash -l
#SBATCH --time=12:00:00
#SBATCH --output=output.out
#SBATCH --mem=128G
#SBATCH --cpus-per-task=64


module load mamba
source activate environment
srun python src/initialize_replay_buffer_with_lichess.py
