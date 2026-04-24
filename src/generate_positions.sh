#!/bin/bash -l
#SBATCH --time=03:00:00
#SBATCH --output=generations.out
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --account=ellis_users
# #SBATCH --account=aalto_users
#SBATCH --constraint="a100|h100|h200|b300"
# #SBATCH --partition=gpu-debug
#SBATCH --cpus-per-gpu=16


module load mamba
module load triton/2024.1-gcc gcc/12.3.0  # needed for torch.compile
source activate environment
srun python src/generate_positions.py --run_type supervised --run_name no_context_move_model --checkpoint_name model_0700000.pt --temperature 1.0 --steps 512 --output_file no_context_move_positions1.csv
