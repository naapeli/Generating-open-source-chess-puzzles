---
license: mit
tags:
- diffusers
- chess
- custom-pipeline
pipeline_tag: text-generation
---

# Chess Puzzle Generator

A masked diffusion model for generating chess puzzles conditioned on themes, ratings, moves and partial boards. This model is not guaranteed to generate a puzzle and the generations should be filtered afterwards.

## Usage

```python
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "naapeli/chess-puzzle-generator",
    trust_remote_code=True,
)
# For exactly the same model as in the paper, use revision="paper":
# pipeline = DiffusionPipeline.from_pretrained(
#     "naapeli/chess-puzzle-generator",
#     revision="paper",
#     trust_remote_code=True,
# )

results = pipeline(
    themes="mateIn2 middlegame",
    rating=1800,
    batch_size=1,
    steps=64,
    device=device,
)
print(results[0])

# To condition the model on a partial board and a best move, use the following:
partial_fen = "?????rk?/?????ppp/????????/????????/????????/???B????/????????/???????? w ??-- - ? ?"
best_move = "d3h7"
results_partial = pipeline(
    themes="mate",
    rating=1600,
    partial_board=partial_fen,
    best_move=best_move,
    batch_size=1,
    steps=256,
    device=device,
)

```
