#!/bin/bash -l
#SBATCH --time=0-12:00:00
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
srun python src/train_rl_espo.py --run_name run8 --reference_path "./src/runs/supervised/no_context_move_model/model_0600000.pt" --batch_size 16 --n_gradient_updates_per_generation 1 --beta 0.01 --steps 128 --lr 1e-6 --n_arteficial 16
