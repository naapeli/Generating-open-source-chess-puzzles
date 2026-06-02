import argparse
import queue
import torch
from pathlib import Path
import pandas as pd
from chess.engine import SimpleEngine
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
import os

from MaskedDiffusion.model import MaskedDiffusion
from RatingModel.model import RatingModel
from rl.espo import generate_random_themes, theme_reward
from tokenization.tokenization import theme_preprocessor, scale_ratings, tokens_to_fen, tokens_to_move, unscale_ratings
from metrics.themes import legal, get_unique_puzzle_from_fen, counter_intuitive
from metrics.cook import cook
from MaskingSchedule.MaskingSchedule import string_to_schedule


torch.set_float32_matmul_precision("high")

parser = argparse.ArgumentParser()
parser.add_argument("--run_type", choices=["supervised", "rl"], required=True)
parser.add_argument("--checkpoint_name", type=str, default=None)
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--steps", type=int, default=512)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--context_dataset", choices=["train", "test"], default="test")
parser.add_argument("--generate_move_last", action="store_true")
parser.add_argument("--n_fens", type=int, default=10_000)
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
base_path = Path("./src")
checkpoint = torch.load(base_path / "runs" / args.run_type / args.run_name / args.checkpoint_name, map_location="cpu", weights_only=False)

config = checkpoint["config"]
# config.schedule = "linear"
# config.masking_schedule = string_to_schedule(config.schedule)
model = MaskedDiffusion(config)
model.load_state_dict(checkpoint["model"])
model.to(device=device)

results_list = []

n = args.n_fens
batch_size = 1024  # 8192


n_jobs = int(os.getenv("SLURM_CPUS_PER_GPU")) - 2
stockfish_path = base_path / ".." / "Stockfish" / "src" / "stockfish"

engine_pool = queue.Queue()
for _ in range(n_jobs):
    engine = SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Threads": 1, "Hash": 32})
    engine_pool.put(engine)

if config.use_context:
    dataset_name = "trainset.pt" if args.context_dataset == "train" else "testset.pt"
    dataset = torch.load(base_path / "dataset" / "with_best_move" / dataset_name, weights_only=False, map_location="cpu")
else:
    dataset = None

def process_puzzle(fen_tokens, move_tokens, base_theme, base_rating, device):
    entry = {"target_themes": base_theme, "target_rating": base_rating, "fen": None, "best_move": None, "is_legal": False, "is_puzzle": False, "counter_intuitive": None, "actual_themes": None, "themes_match": None, "main_line": None}

    engine = engine_pool.get()

    try:
        try:
            fen = tokens_to_fen(fen_tokens)
            entry["fen"] = fen
            if move_tokens is not None:
                move = tokens_to_move(move_tokens)
                entry["best_move"] = move
        except:
            return entry

        if not legal(fen):
            return entry
        
        entry["is_legal"] = True
        
        engine.configure({"Clear Hash": None})
        entry["counter_intuitive"] = counter_intuitive(fen, engine)
        puzzle = get_unique_puzzle_from_fen(fen, engine)
        
        if puzzle is not None:
            entry["is_puzzle"] = True
            entry["main_line"] = " ".join([move.uci() for move in puzzle.mainline])
            existing_themes = cook(puzzle, engine)
            entry["actual_themes"] = existing_themes
            
            existing_themes_one_hot = torch.from_numpy(theme_preprocessor.transform([existing_themes])).to(device=device, dtype=torch.float32)
            
            if config.use_context:
                entry["themes_match"] = theme_reward(base_theme, existing_themes)

        return entry

    finally:
        engine_pool.put(engine)


for iteration in range(n // batch_size + 1):
    print(iteration + 1, n // batch_size + 1, flush=True)
    if config.use_context:
        indices = torch.randint(0, len(dataset), (batch_size,)).tolist()
        sampled_items = [dataset[idx] for idx in indices]
        
        themes_one_hot = torch.stack([torch.as_tensor(item[2]) for item in sampled_items]).to(device=device, dtype=torch.float32)
        scaled_ratings = torch.stack([torch.as_tensor(item[3]) for item in sampled_items]).to(device=device, dtype=torch.float32)
        
        base_themes = theme_preprocessor.inverse_transform(themes_one_hot.cpu().numpy())
        base_ratings = unscale_ratings(scaled_ratings).tolist()
    else:
        themes_one_hot = None
        scaled_ratings = None
        base_themes = None
        base_ratings = None

    module = model.module if hasattr(model, "module") else model
    start = perf_counter()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        tokens = module.sample(themes_one_hot, scaled_ratings, batch_size=batch_size, steps=args.steps, temperature=args.temperature, generate_move_last=args.generate_move_last)
    print("Sampling time:", perf_counter() - start, flush=True)
    
    if config.predict_moves:
        fen_tokens = tokens[:, :config.fen_length]
        move_tokens = tokens[:, config.fen_length:]
    else:
        fen_tokens = tokens
        move_tokens = None

    start2 = perf_counter()
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        args_list = [
            (
                fen_tokens[i],
                move_tokens[i] if move_tokens is not None else None,
                base_themes[i] if themes_one_hot is not None else None,
                base_ratings[i] if scaled_ratings is not None else None,
                device
            ) for i in range(batch_size)
        ]
        batch_results = list(executor.map(lambda p: process_puzzle(*p), args_list))
    print("Processing time:", perf_counter() - start2, flush=True)
    print("Total time:", perf_counter() - start, flush=True)
    
    results_list.extend(batch_results)

while not engine_pool.empty():
    engine = engine_pool.get()
    engine.quit()

df = pd.DataFrame(results_list)
output_path = base_path / "Generate_positions" / args.output_file
df.to_csv(output_path, index=False)
