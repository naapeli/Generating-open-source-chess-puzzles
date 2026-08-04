#!/bin/bash -l
#SBATCH --time=16:00:00
#SBATCH --output=generations.out
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --account=ellis_users
#SBATCH --constraint="h200"
# #SBATCH --constraint="volta"
#SBATCH --cpus-per-task=64


module load mamba
module load triton/2024.1-gcc gcc/12.3.0  # needed for torch.compile
source activate environment
# srun python src/generate_positions.py --run_type supervised --run_name final_model --checkpoint_name model_1000000.pt --n_fens 200000 --batch_size 32768 --temperature 1.0 --steps 256 --output_file final_model/supervised/generated_positions/steps64.csv --context_dataset train #--generate_move_last
srun python src/generate_positions.py --run_type rl --run_name final_large_runs --checkpoint_name final_thesis_experiments/full_diversity11/model_0003500.pt --n_fens 1000000 --batch_size 32768 --temperature 1.0 --steps 16 --output_file final_model/rl/full_diversity11/steps_experiment_own_theme_distribution/steps16v2.csv --context_dataset random #--generate_move_last
