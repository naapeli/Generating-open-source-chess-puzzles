# Generating Open-Source Chess Puzzles

This repository contains the official code for our AAAI 2027 submission on `Conditional Generation of Creative Chess Puzzles with Diffusion Models`.

## Overview

Procedurally generating high-quality, aesthetic, and challenging chess puzzles is a complex task. Traditional methods rely on filtering positions from human play. In this project, we train a generative model for chess puzzle generation. Furthermore, we employ Reinforcement Learning via Denoising Diffusion Policy Optimization (DDPO) for further model training.

### Key Contributions
- **Masked Diffusion for Chess:** A novel application of discrete diffusion models for generating chess positions.
- **RL-based Alignment (DDPO):** Utilizing Stockfish evaluations and theme heuristics as reward signals to steer the diffusion process towards better puzzles.
- **Controllable Generation:** Generating puzzles conditional on themes, ratings, or partial board states.

---

## Environment Setup

The environment requirements are specified in `environment.yml`.

To set up the environment, run:

```bash
conda env create -f environment.yml
conda activate environment
```

**Note:** You must have the [Stockfish engine](https://stockfishchess.org/download/) installed and accessible, or compiled inside the project directory structure (`../Stockfish/src/stockfish` by default).

---

## Project Structure

- `src/MaskedDiffusion/`: Contains the architecture for the Masked Diffusion model.
- `src/MaskingSchedule/`: Implementations of different masking schedules (linear, cosine, etc.) for the diffusion process.
- `src/rl/`: Implementation of the theme based reward.
- `src/metrics/`: Scripts for evaluating generated puzzles, detecting themes, and calculating rewards.
- `src/tokenization/`: Custom tokenizers for converting FEN strings and chess moves into discrete tokens suitable for the model.
- `src/Config.py`: Centralized configuration dataclass for model hyperparameters and training settings.

---

## Usage

### 1. Supervised Training

To train the base Masked Diffusion model on a dataset of chess puzzles on an HPC:

```bash
sbatch src/supervised.sh
```

### 2. Reinforcement Learning (DDPO)

To align a pre-trained supervised model using DDPO:

```bash
sbatch src/train_rl_ddpo.sh
```

### 3. Evaluation

The repository includes several scripts for evaluating checkpoints, measuring distances to the training set, and comparing generated puzzles to the Lichess database.

- **Checkpoint Evaluation:**
  ```bash
  sbatch src/evaluate_checkpoints.sh
  ```
- **Diffusion Trajectories:**
  ```bash
  python compute_diffusion_trajectories.py
  sbatch src/compute_diffusion_trajectories.sh
  ```
- **Diffusion Trajectories:**
  ```bash
  sbatch src/compute_diffusion_trajectories.sh
  ```
- **Conditioning experiments:**
  ```bash
  sbatch src/generate_from_partial_board.sh
  ```


## License
MIT License
