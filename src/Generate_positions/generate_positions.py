import torch
import numpy as np
from pathlib import Path
import pandas as pd
from chess.engine import SimpleEngine

from MaskedDiffusion.model import MaskedDiffusion
from RatingModel.model import RatingModel
from rl.espo import generate_random_themes, theme_reward
from tokenization.tokenization import theme_preprocessor, scale_ratings, tokens_to_fen, unscale_ratings
from metrics.themes import legal, get_unique_puzzle_from_fen, counter_intuitive
from metrics.cook import cook


torch.set_float32_matmul_precision('high')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
base_path = Path("./../src")
checkpoint = torch.load(base_path / "supervised_checkpoints" / "model_0760000.pt", map_location="cpu", weights_only=False)

config = checkpoint["config"]
model = MaskedDiffusion(config)
model.load_state_dict(checkpoint["model"])
model.to(device=device)

rating_model_checkpoint = torch.load(base_path / "rating_model_checkpoints" / "model_0063000.pt", map_location="cpu", weights_only=False)
rating_model = RatingModel(rating_model_checkpoint["config"])
rating_model.load_state_dict(rating_model_checkpoint["model"])
rating_model.to(device=device)

results_list = []

n = 10000
batch_size = 256

engine = SimpleEngine.popen_uci(base_path / ".." / "Stockfish" / "src" / "stockfish")

random_themes_and_ratings = False
if not random_themes_and_ratings:
    dataset = pd.read_csv(base_path / "dataset" / "dataset.csv", nrows=100 * n)
    dataset_themes = dataset["Themes"].str.split().tolist()
    dataset_ratings = torch.from_numpy(dataset["Rating"].to_numpy())

for iteration in range(n // batch_size):
    print(iteration + 1, flush=True)
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
    fen_tokens = module.sample(themes_one_hot, scaled_ratings, steps=128)

    for i, tokens in enumerate(fen_tokens):
        entry = {
            "target_themes": base_themes[i],
            "target_rating": base_ratings[i].item(),
            "fen": None,
            "is_legal": False,
            "is_puzzle": False,
            "counter_intuitive": None,
            "actual_themes": None,
            "predicted_rating": None,
            "themes_match": None
        }

        try:
            fen = tokens_to_fen(tokens)
            entry["fen"] = fen
        except:
            results_list.append(entry)
            continue

        if not legal(fen):
            results_list.append(entry)
            continue
        
        entry["is_legal"] = True
        
        puzzle = get_unique_puzzle_from_fen(fen, engine)
        entry["counter_intuitive"] = counter_intuitive(fen, engine)
        
        if puzzle is not None:
            entry["is_puzzle"] = True
            existing_themes = cook(puzzle, engine)
            entry["actual_themes"] = existing_themes
            
            existing_themes_one_hot = torch.from_numpy(theme_preprocessor.transform([existing_themes])).to(device=device, dtype=torch.float32)
            
            with torch.no_grad():
                puzzle_rating = rating_model(tokens.unsqueeze(0), existing_themes_one_hot)
            
            entry["predicted_rating"] = unscale_ratings(puzzle_rating).item()
            if random_themes_and_ratings:
                entry["themes_match"] = theme_reward(base_themes[i], existing_themes)
            else:
                entry["themes_match"] = len(set(base_themes[i]).intersection(existing_themes)) > len(base_themes[i]) // 2  # if over half of the themes match

        results_list.append(entry)

df = pd.DataFrame(results_list)
output_path = base_path / "generated_puzzles_lichess_dataset_theme_distribution_correct_ratings.csv"
df.to_csv(output_path, index=False)
engine.quit()
