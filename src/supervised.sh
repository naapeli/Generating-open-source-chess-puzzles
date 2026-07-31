#!/bin/bash -l
#SBATCH --job-name=MaskedDiffusion
#SBATCH --time=0-72:00:00
#SBATCH --output=output.out
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --constraint="h200|b300"
#SBATCH --ntasks=8
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=16


module load mamba
module load triton/2024.1-gcc gcc/12.3.0  # needed for torch.compile
source activate environment
srun python src/supervised_training.py --distributed --run_name final_model_no_move 
# --checkpoint_name model_1000000.pt
