#!/bin/bash -l
#SBATCH --time=0-48:00:00
# #SBATCH --time=0-00:30:00
#SBATCH --output=outputrl.out
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --account=ellis_users
# #SBATCH --gres=min-vram:32g
# #SBATCH --partition=gpu-v100-32g
#SBATCH --constraint="h100|h200"  # a100|
# #SBATCH --partition=gpu-debug
#SBATCH --cpus-per-task=16


module load mamba
module load triton/2024.1-gcc gcc/12.3.0  # needed for torch.compile
source activate environment
srun python src/train_rl_espo.py --run_name no_conditioning/run1 --reference_path "./src/runs/supervised/no_context_move_model/model_0600000.pt" --batch_size 32 --n_gradient_updates_per_generation 2 --beta 0.01 --steps 128 --lr 1e-6 --n_arteficial 32 --save_period 500   # --distributed
