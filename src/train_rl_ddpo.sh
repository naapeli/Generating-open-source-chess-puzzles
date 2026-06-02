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
#SBATCH --cpus-per-task=64


module load mamba
module load triton/2024.1-gcc gcc/12.3.0  # needed for torch.compile
source activate environment
srun python src/train_rl_ddpo.py --run_name final_large_runs/ddpo/Ablations/all_diversity_filtering_larger_entropy_bonus --reference_path "./src/runs/supervised/final_model/model_1000000.pt" --batch_size 16 --group_size 6 --ppo_epochs 2 --ppo_minibatch_size 1536 --kl_coef 0.03 --entropy_coef 0.03 --steps 128 --lr 3e-5 --n_artificial 0 --sup_loss_coef 0.0
