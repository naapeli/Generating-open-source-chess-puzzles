#!/bin/bash -l
#SBATCH --time=0-04:00:00
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
srun python src/train_rl_ddpo.py --run_name paper_experiments/fig9v5 --reference_path "./src/runs/supervised/no_context_move_model/model_0600000.pt" --batch_size 64 --ppo_epochs 2 --ppo_minibatch_size 256 --kl_coef 0.0 --entropy_coef 0.0 --steps 128 --lr 3e-5 --n_artificial 0  # 16
