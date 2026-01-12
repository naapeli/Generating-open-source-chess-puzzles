#!/bin/bash -l
#SBATCH --time=00:10:00
#SBATCH --gpus=1

module load scicomp-python-env
python supervised_training.py
