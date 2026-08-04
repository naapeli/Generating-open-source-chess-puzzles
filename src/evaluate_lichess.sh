#!/bin/bash -l
#SBATCH --time=04:00:00
#SBATCH --output=evaluate_lichess.out
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

module load mamba
source activate environment
# srun python src/evaluate_lichess.py --n_puzzles 10000000 --input_file Generate_positions/final_model/rl/training_progress/test_no_move_last/ --output_file final_model/rl/training_progress/test_no_move_lastv2/
srun python src/evaluate_lichess.py --n_puzzles 200000 --input_file dataset/dataset.csv --output_file Lichess/lichess_xidong.csv
