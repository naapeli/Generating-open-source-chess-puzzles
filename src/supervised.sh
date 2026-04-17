#!/bin/bash -l
#SBATCH --job-name=MaskedDiffusion
#SBATCH --time=0-24:00:00
#SBATCH --output=output.out
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --constraint="a100|h100|h200|b300"
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=10


module load mamba
module load triton/2024.1-gcc gcc/12.3.0  # needed for torch.compile
source activate environment
srun python src/supervised_training.py --distributed --run_name best_move_model --checkpoint_name model_0980000.pt
