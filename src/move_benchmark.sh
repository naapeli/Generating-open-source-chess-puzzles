#!/bin/bash -l
#SBATCH --time=03:00:00
#SBATCH --output=move_benchmark.out
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint="a100|h100|h200|b300"
# #SBATCH --partition=gpu-debug
#SBATCH --cpus-per-gpu=16


module load mamba
module load triton/2024.1-gcc gcc/12.3.0  # needed for torch.compile
source activate environment
srun python src/move_benchmark.py --checkpoint src/runs/supervised/best_move_model/model_0980000.pt --batch_size 12288 --steps 32 --n_samples 50000  # --puzzles
