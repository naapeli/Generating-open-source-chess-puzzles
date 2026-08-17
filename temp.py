from diffusers import DiffusionPipeline
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    steps=4,
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
    steps=4,
    device=device,
)
print(results_partial)
