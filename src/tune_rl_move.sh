#!/bin/bash -l
#SBATCH --time=0-48:00:00
# #SBATCH --time=0-00:30:00
#SBATCH --output=outputrl.out
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --account=ellis_users
#SBATCH --constraint="h100|h200"  # a100|
# #SBATCH --partition=gpu-debug
#SBATCH --cpus-per-task=16


module load mamba
module load triton/2024.1-gcc gcc/12.3.0  # needed for torch.compile
source activate environment
srun python src/tune_rl_move.py --n_trials 50 --study_name tune_move_no_context1 --steps_per_trial 2000 --reward_structure "rewards = torch.where(pass_diversity_filtering & move_matches, 1.0, 0.0)" --model_path no_context_move_model/model_0600000.pt
