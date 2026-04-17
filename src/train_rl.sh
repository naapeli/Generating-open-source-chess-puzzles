#!/bin/bash -l
#SBATCH --time=0-48:00:00
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
srun python src/train_rl.py --run_name run2 --reference_path "./src/runs/supervised/real_model_v2/model_0940000.pt" --lr 1e-6 --beta 1e-1 --group_size 4 --batch_size 8 --temperature 1e-2 --reward_structure "rewards = legal_position * pass_diversity_filtering * (unique_solution + counter_intuitive_solution)" --n_gradient_updates_per_generation 1 --save_period 500
