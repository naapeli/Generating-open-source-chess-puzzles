#!/bin/bash -l
#SBATCH --time=0-24:00:00
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
srun python src/train_rl.py --run_name test3 --reference_path "./src/runs/supervised/no_context_move_model/model_0600000.pt" --lr 1e-7 --beta 100 --group_size 16 --batch_size 2 --temperature 1 --reward_structure "rewards = torch.where(legal_position & unique_solution & counter_intuitive_solution, 1.0, 0.0); rewards = torch.where(legal_position, rewards, -2.0)" --n_gradient_updates_per_generation 1 --save_period 500 --weight_decay 0
