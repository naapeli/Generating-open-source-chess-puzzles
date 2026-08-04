#!/bin/bash -l
#SBATCH --time=06:00:00
#SBATCH --output=evaluate_checkpoints.out
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --account=ellis_users
#SBATCH --constraint="h200"
#SBATCH --cpus-per-gpu=64


module load mamba
module load triton/2024.1-gcc gcc/12.3.0
source activate environment

# CHECKPOINT_DIR=${CHECKPOINT_DIR:-"src/runs/supervised/final_model_no_move"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"src/runs/rl/final_large_runs/final_thesis_experiments/full_diversity11"}
# OUTPUT_DIR=${OUTPUT_DIR:-"src/Generate_positions/final_model/supervised/training_progress/train_context"}
OUTPUT_DIR=${OUTPUT_DIR:-"src/Generate_positions/final_model/rl/training_progress/test_no_move_lastv3"}
N_FENS=${N_FENS:-30000}
TEMPERATURE=${TEMPERATURE:-1.0}
STEPS=${STEPS:-256}
CONTEXT_DATASET=${CONTEXT_DATASET:-"train"}

srun python src/evaluate_checkpoints.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --n_fens "$N_FENS" \
    --temperature "$TEMPERATURE" \
    --steps "$STEPS" \
    --context_dataset "$CONTEXT_DATASET"
    # --generate_move_last
