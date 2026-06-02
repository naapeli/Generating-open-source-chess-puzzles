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
#SBATCH --cpus-per-task=32


module load mamba
module load triton/2024.1-gcc gcc/12.3.0  # needed for torch.compile
source activate environment
# srun python src/train_rl_espo.py --run_name no_conditioning/run5 --reference_path "./src/runs/supervised/no_context_move_model/model_0600000.pt" --batch_size 64 --n_gradient_updates_per_generation 8 --beta 0.03 --steps 128 --lr 3e-6 --n_arteficial 64 --save_period 500 --checkpoint_model model_0001500.pt

# srun python src/train_rl_espo.py --run_name conditioning_with_move/run4 --reference_path "./src/runs/supervised/best_move_model/model_0960000.pt" --batch_size 64 --n_gradient_updates_per_generation 8 --beta 0.03 --steps 128 --lr 3e-6 --n_arteficial 16 --save_period 250  --checkpoint_model model_0001000.pt

# srun python src/train_rl_espo.py --run_name test/run8 --reference_path "./src/runs/supervised/best_move_model/model_0960000.pt" --batch_size 64 --n_gradient_updates_per_generation 8 --beta 0.03 --steps 128 --lr 3e-6 --n_arteficial 0 --save_period 500

srun python src/train_rl_espo.py --run_name Ablations/no_diversity --reference_path "./src/runs/supervised/final_model/model_1000000.pt" --batch_size 24 --group_size 6 --n_gradient_updates_per_generation 2 --beta 0.0 --entropy_coef 0.0 --steps 128 --lr 3e-5 --n_arteficial 0 --gamma 0.0 --save_period 50000


# Notes:
# good gamma probably around 0.1
# good n_arteficial probably around 256
