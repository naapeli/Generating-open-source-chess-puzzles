import queue
import torch
import numpy as np
from pathlib import Path
import pandas as pd
from chess.engine import SimpleEngine
from joblib import Parallel, delayed

from MaskedDiffusion.model import MaskedDiffusion
from RatingModel.model import RatingModel
from rl.espo import generate_random_themes, theme_reward
from tokenization.tokenization import theme_preprocessor, scale_ratings, tokens_to_fen, tokens_to_move, unscale_ratings
from metrics.themes import legal, get_unique_puzzle_from_fen, counter_intuitive
from metrics.cook import cook


torch.set_float32_matmul_precision("high")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
base_path = Path("./src")
# checkpoint = torch.load(base_path / "supervised_checkpoints" / "model_0940000.pt", map_location="cpu", weights_only=False)
# checkpoint = torch.load(base_path / "rl_checkpoints" / "model_0007000.pt", map_location="cpu", weights_only=False)
checkpoint = torch.load(base_path / "supervised_checkpoints" / "best_move_model" / "model_0680000.pt", map_location="cpu", weights_only=False)

config = checkpoint["config"]
model = MaskedDiffusion(config)
model.load_state_dict(checkpoint["model"])
model.to(device=device)

rating_model_checkpoint = torch.load(base_path / "rating_model_checkpoints" / "model_0063000.pt", map_location="cpu", weights_only=False)
rating_model = RatingModel(rating_model_checkpoint["config"])
rating_model.load_state_dict(rating_model_checkpoint["model"])
rating_model.to(device=device)

results_list = []

n = 10_000
batch_size = 1024  # 8192


n_jobs = 16
stockfish_path = base_path / ".." / "Stockfish" / "src" / "stockfish"

engine_pool = queue.Queue()
for _ in range(n_jobs):
    engine_pool.put(SimpleEngine.popen_uci(stockfish_path))

random_themes_and_ratings = True
if not random_themes_and_ratings:
    dataset = pd.read_csv(base_path / "dataset" / "dataset.csv", nrows=1000 * n)
    dataset_themes = dataset["Themes"].str.split().tolist()
    dataset_ratings = torch.from_numpy(dataset["Rating"].to_numpy())

def process_puzzle(fen_tokens, move_tokens, base_theme, base_rating, device, rating_model):
    entry = {"target_themes": base_theme, "target_rating": base_rating.item(), "fen": None, "best_move": None, "is_legal": False, "is_puzzle": False, "counter_intuitive": None, "actual_themes": None, "predicted_rating": None, "themes_match": None, "main_line": None, "best_move_matches": None}

    engine = engine_pool.get()

    try:
        try:
            fen = tokens_to_fen(fen_tokens)
            move = tokens_to_move(move_tokens)
            entry["fen"] = fen
            entry["best_move"] = move
        except:
            return entry

        if not legal(fen):
            return entry
        
        entry["is_legal"] = True
        
        puzzle = get_unique_puzzle_from_fen(fen, engine)
        entry["counter_intuitive"] = counter_intuitive(fen, engine)
        
        if puzzle is not None:
            entry["is_puzzle"] = True
            entry["main_line"] = " ".join([move.uci() for move in puzzle.mainline])
            entry["best_move_matches"] = move == puzzle.mainline[0].uci()
            existing_themes = cook(puzzle, engine)
            entry["actual_themes"] = existing_themes
            
            existing_themes_one_hot = torch.from_numpy(theme_preprocessor.transform([existing_themes])).to(device=device, dtype=torch.float32)
            
            with torch.no_grad():
                puzzle_rating = rating_model(fen_tokens.unsqueeze(0), existing_themes_one_hot)
            
            entry["predicted_rating"] = unscale_ratings(puzzle_rating).item()
            
            entry["themes_match"] = theme_reward(base_theme, existing_themes)

        return entry

    finally:
        engine_pool.put(engine)


for iteration in range(n // batch_size):
    print(iteration + 1, n // batch_size, flush=True)
    if random_themes_and_ratings:
        base_themes, base_ratings = generate_random_themes(batch_size)
    else:
        indices = torch.randint(0, len(dataset_ratings), (batch_size,))
        base_ratings = dataset_ratings[indices]
        base_themes = [dataset_themes[index] for index in indices]
    
    base_ratings = base_ratings.to(device=device, dtype=torch.float32)
    themes_one_hot = torch.from_numpy(theme_preprocessor.transform(base_themes)).to(device=device, dtype=torch.float32)
    scaled_ratings = scale_ratings(base_ratings).to(device=device, dtype=torch.float32)

    module = model.module if hasattr(model, "module") else model
    tokens = module.sample(themes_one_hot, scaled_ratings, steps=128)
    fen_tokens = tokens[:, :config.fen_length]
    move_tokens = tokens[:, config.fen_length:]

    batch_results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(process_puzzle)(fen_tokens[i], move_tokens[i], base_themes[i], base_ratings[i], device, rating_model) for i in range(batch_size)
    )
    
    results_list.extend(batch_results)

while not engine_pool.empty():
    engine = engine_pool.get()
    engine.quit()

df = pd.DataFrame(results_list)
output_path = base_path / "Generate_positions" / "best_move_model.csv"
df.to_csv(output_path, index=False)
