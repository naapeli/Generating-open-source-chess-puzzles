#!/bin/bash -l
#SBATCH --time=01:00:00
#SBATCH --output=compute_distances.out
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32


module load mamba
source activate environment

srun python src/compute_distances.py \
  --generated_csv src/Generate_positions/final_model/rl/full_diversity18/checkpoint_20000 \
  --lichess_csv src/dataset/dataset.csv \
  --self_sample_size 40000 \
  --lichess_sample_size 100000 \
  --chunk_size 10000
