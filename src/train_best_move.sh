#!/bin/bash -l
#SBATCH --time=0-03:00:00
#SBATCH --output=output_best_move.out
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --account=ellis_users
#SBATCH --constraint="h100|h200"  # a100|
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=10


module load mamba
module load triton/2024.1-gcc gcc/12.3.0  # needed for torch.compile
source activate environment
srun python src/train_best_move.py --distributed --run_name run2 --starting_model_path runs/supervised/no_context_move_model/model_0600000.pt
# srun python src/train_best_move.py --distributed --run_name run2 --starting_model_path runs/supervised/best_move_model/model_0980000.pt
