#!/bin/bash -l
#SBATCH --time=00:10:00
#SBATCH --output=compute_distances.out
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32


module load mamba
source activate environment

srun python src/compute_distances.py \
  --lichess_csv src/Generate_positions/Lichess/lichess_xidong_large.csv \
  --generated_csv src/Generate_positions/final_model/rl/training_progress/test_no_move_lastv2/ \
  --self_sample_size 10000 \
  --lichess_sample_size 100000 \
  --chunk_size 10000
