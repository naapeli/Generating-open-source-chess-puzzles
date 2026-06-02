#!/bin/bash -l
#SBATCH --job-name=MaskedDiffusion
#SBATCH --time=0-48:00:00
#SBATCH --output=output.out
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --account=ellis_users
#SBATCH --constraint="h100|h200|b300"  # a100|
#SBATCH --ntasks=8
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=16


module load mamba
module load triton/2024.1-gcc gcc/12.3.0  # needed for torch.compile
source activate environment
srun python src/supervised_training.py --distributed --run_name final_model --checkpoint_name model_1000000.pt
