#!/bin/bash -l
#SBATCH --time=06:00:00
#SBATCH --output=evaluate_checkpoints.out
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
# #SBATCH --account=ellis_users
#SBATCH --constraint="a100|h100|h200"
#SBATCH --cpus-per-gpu=64


module load mamba
module load triton/2024.1-gcc gcc/12.3.0
source activate environment

CHECKPOINT_DIR=${CHECKPOINT_DIR:-"src/runs/supervised/final_model"}
OUTPUT_DIR=${OUTPUT_DIR:-"src/Generate_positions/final_model/supervised/training_progress/test_no_move_last"}
N_FENS=${N_FENS:-10000}
TEMPERATURE=${TEMPERATURE:-1.0}
STEPS=${STEPS:-256}
CONTEXT_DATASET=${CONTEXT_DATASET:-"test"}

srun python src/evaluate_checkpoints.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --n_fens "$N_FENS" \
    --temperature "$TEMPERATURE" \
    --steps "$STEPS" \
    --context_dataset "$CONTEXT_DATASET"
    # --generate_move_last
