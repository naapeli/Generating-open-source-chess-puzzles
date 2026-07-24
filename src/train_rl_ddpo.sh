#!/bin/bash -l
#SBATCH --time=0-120:00:00
#SBATCH --output=outputrl.out
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint="h200"
#SBATCH --cpus-per-task=32


module load mamba
module load triton/2024.1-gcc gcc/12.3.0  # needed for torch.compile
source activate environment
srun python src/train_rl_ddpo.py --run_name run1 --reference_path "./src/runs/supervised/model_1000000.pt" --batch_size 6 --group_size 16 --ppo_epochs 1 --ppo_minibatch_size 1536 --kl_coef 0.1 --entropy_coef 0.2 --steps 128 --lr 3e-5 --n_artificial 0 --sup_loss_coef 0.0 --save_period 2000 --n_generations 100000
