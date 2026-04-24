#!/bin/bash -l
#SBATCH --job-name=MaskedDiffusion
#SBATCH --time=0-12:00:00
#SBATCH --output=output.out
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --constraint="h100|h200|b300"  # a100|
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=10


module load mamba
module load triton/2024.1-gcc gcc/12.3.0  # needed for torch.compile
source activate environment
srun python src/supervised_training.py --distributed --run_name no_context_move_model --checkpoint_name model_0600000.pt
