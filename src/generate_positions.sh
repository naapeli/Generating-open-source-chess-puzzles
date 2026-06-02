#!/bin/bash -l
#SBATCH --time=12:00:00
#SBATCH --output=generations.out
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --account=ellis_users
#SBATCH --constraint="a100|h100|h200"
# #SBATCH --partition=gpu-debug
#SBATCH --cpus-per-gpu=32


module load mamba
module load triton/2024.1-gcc gcc/12.3.0  # needed for torch.compile
source activate environment
srun python src/generate_positions.py --run_type supervised --run_name final_model --checkpoint_name model_1000000.pt --n_fens 200000 --temperature 1.0 --steps 256 --output_file final_model/supervised/v4/test_context_move_last.csv --context_dataset test --generate_move_last
