#!/bin/bash -l
#SBATCH --time=00:10:00
#SBATCH --output=temp.out
#SBATCH --mem=64G
#SBATCH --nodes=1


module load mamba
source activate environment
srun python src/Generate_positions/visualize_puzzles.py
