#!/bin/bash -l
#SBATCH --time=03:00:00
#SBATCH --output=evaluate_lichess.out
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

module load mamba
source activate environment
srun python src/evaluate_lichess.py --n_puzzles 10000 --output_file Lichess/lichess_first_time_move_found_division_50.csv
