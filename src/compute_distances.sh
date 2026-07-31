#!/bin/bash -l
#SBATCH --time=00:10:00
#SBATCH --output=compute_distances.out
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16


module load mamba
source activate environment

srun python src/compute_distances.py \
  --lichess_csv src/Generate_positions/Lichess/lichess.csv \
  --self_sample_size 40000 \
  --lichess_sample_size 100000 \
  --chunk_size 40000

# --generated_csv src/Generate_positions/final_model/rl/full_diversity18/checkpoint_20000 \
