#!/bin/bash -l
#SBATCH --time=72:00:00
#SBATCH --output=generations.out
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
# #SBATCH --ntasks=8
# #SBATCH --gpus=8
#SBATCH --account=ellis_users
#SBATCH --constraint="h200"
# #SBATCH --partition=gpu-debug
#SBATCH --cpus-per-gpu=128


module load mamba
module load triton/2024.1-gcc gcc/12.3.0  # needed for torch.compile
source activate environment
# srun python src/generate_positions.py --run_type supervised --run_name final_model_no_move --checkpoint_name model_1000000.pt --n_fens 1000000 --batch_size 32768 --temperature 1.0 --steps 256 --output_file final_model_no_move/supervised/checkpoint1000000/test_context.csv --context_dataset test #--generate_move_last
srun python src/generate_positions.py --run_type rl --run_name final_large_runs --checkpoint_name final_thesis_experiments/full_diversity18/model_0020000.pt --n_fens 1000000 --batch_size 32768 --temperature 1.0 --steps 256 --output_file final_model/rl/full_diversity18/checkpoint_20000/test_context_no_move_last.csv --context_dataset test #--generate_move_last
