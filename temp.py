from diffusers import DiffusionPipeline
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = "model_main"
# path = "naapeli/chess-puzzle-generator"
pipeline = DiffusionPipeline.from_pretrained(
    path,
    trust_remote_code=True,
)
# For exactly the same model as in the paper, use revision="paper":
# pipeline = DiffusionPipeline.from_pretrained(
#     "naapeli/chess-puzzle-generator",
#     revision="paper",
#     trust_remote_code=True,
# )
pipeline.to(device)

themes = pipeline.Theme
schedules = pipeline.Schedule

results = pipeline(
    themes=[themes.mateIn2, themes.middlegame],
    rating=1800,
    batch_size=1,
    steps=4,
    schedule=schedules.linear,
)
print("Position 1:", results[0].fen, results[0].move)

# To condition the model on a partial board and a best move, use the following:
partial_fen = "?????rk?/?????ppp/????????/????????/????????/???B????/????????/???????? w ??-- - ? ?"
best_move = "d3h7"
results_partial = pipeline(
    themes=themes.mate,
    rating=1600,
    partial_board=partial_fen,
    best_move=best_move,
    batch_size=1,
    schedule=schedules.cosine,
    steps=4,
)
print("Position 2:", results_partial[0].fen, results_partial[0].move)

results = pipeline(
    themes=[themes.middlegame, themes.veryLong, themes.fork, themes.sacrifice],
    rating=2000,
    batch_size=4,
    steps=16,
)
for i, pos in enumerate(results, start=1):
    print(f"Batch {i}: {pos.fen} | move: {pos.move}")

