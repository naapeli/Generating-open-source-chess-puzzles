import torch
from diffusers import DiffusionPipeline

# 1. Load the pipeline from the exported directory
# Note: trust_remote_code=True loads pipeline.py and model.py from the export folder
export_path = "src/test_hf_export"  # or "./test_hf_export" depending on where you ran the command

print(f"Loading pipeline from: {export_path}...")
pipeline = DiffusionPipeline.from_pretrained(
    export_path,
    trust_remote_code=True,
    torch_dtype=torch.float32,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline.to(device)
print(f"Pipeline successfully loaded on device: {device}\n")

# 2. Test standard puzzle generation
print("--- Test 1: Generate puzzles from themes & rating ---")
results = pipeline(
    themes="mateIn2 middlegame",
    rating=1800,
    batch_size=2,
    steps=4,
    schedule="linear",
    device=device,
)

for i, res in enumerate(results):
    print(f"Puzzle {i + 1}:")
    print(f"  FEN:  {res['fen']}")
    print(f"  Move: {res['move']}")

# 3. Test partial board completion
print("\n--- Test 2: Partial board completion ---")
partial_fen = "?????rk?/?????ppp/????????/????????/????????/???B????/????????/???????? w ??-- - ? ?"
best_move = "d3h7"
results_partial = pipeline(
    themes="mate",
    rating=1600,
    partial_board=partial_fen,
    best_move=best_move,
    batch_size=1,
    steps=3,
    device=device,
)
print("Partial Board Result:")
print(f"  FEN:  {results_partial[0]['fen']}")
print(f"  Move: {results_partial[0]['move']}")

# 4. Test alternative schedule
print("\n--- Test 3: Cosine Schedule ---")
results_cosine = pipeline(
    themes="fork short",
    rating=2100,
    batch_size=1,
    steps=4,
    schedule="cosine",
    device=device,
)
print(f"  FEN:  {results_cosine[0]['fen']}")
print(f"  Move: {results_cosine[0]['move']}")
