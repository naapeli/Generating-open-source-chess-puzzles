#!/bin/bash -l
#SBATCH --time=12:00:00
#SBATCH --output=trajectories.out
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --account=ellis_users
#SBATCH --constraint="h200"
#SBATCH --cpus-per-gpu=32

module load mamba
module load triton/2024.1-gcc gcc/12.3.0
source activate environment

python src/compute_diffusion_trajectories.py \
    --run_type rl \
    --run_name final_large_runs/final_thesis_experiments/full_diversity11 \
    --checkpoint_name model_0003500.pt \
    --n_trees 24 \
    --tree_batch_size 4 \
    --steps 256 \
    --n_milestones 6 \
    --branching_factor 4 \
    --temperature 1.0 \
    --output_file final_model/rl/full_diversity11/diffusion_trajectories_short_and_wide.csv \
    --exclude_themes backRankMate mateIn1 oneMove \
    # --target_themes sacrifice

# n_milestones, branching_factor results in branching_factor ** n_milestones leaf nodes per tree
