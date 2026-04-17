#!/bin/bash -l
#SBATCH --time=0-48:00:00
# #SBATCH --time=0-00:30:00
#SBATCH --output=outputrl.out
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint="h100|h200|b300"  # a100|
# #SBATCH --partition=gpu-debug
#SBATCH --cpus-per-task=16


module load mamba
module load triton/2024.1-gcc gcc/12.3.0  # needed for torch.compile
source activate environment
srun python src/tune_rl.py --n_trials 50 --study_name run5 --steps_per_trial 2000 --reward_structure "rewards = torch.where(legal_position & pass_diversity_filtering & unique_solution & counter_intuitive_solution, 1, 0)"
