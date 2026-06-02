import argparse
import queue
import torch
from pathlib import Path
import pandas as pd
from chess.engine import SimpleEngine
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
import os
import re

from MaskedDiffusion.model import MaskedDiffusion
from rl.espo import generate_random_themes, theme_reward
from tokenization.tokenization import theme_preprocessor, scale_ratings, tokens_to_fen, tokens_to_move, unscale_ratings
from metrics.themes import legal, get_unique_puzzle_from_fen, counter_intuitive
from metrics.cook import cook
from MaskingSchedule.MaskingSchedule import string_to_schedule


torch.set_float32_matmul_precision("high")

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory containing checkpoint .pt files")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the evaluated CSVs")
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--steps", type=int, default=512)
parser.add_argument("--context_dataset", choices=["train", "test"], default="test")
parser.add_argument("--generate_move_last", action="store_true")
parser.add_argument("--n_fens", type=int, default=10_000)
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
base_path = Path("./src")

checkpoint_dir = Path(args.checkpoint_dir)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Find all checkpoints
checkpoint_paths = list(checkpoint_dir.glob("*.pt"))
if not checkpoint_paths:
    raise ValueError(f"No checkpoint .pt files found in {checkpoint_dir}")

def get_checkpoint_step(path):
    match = re.search(r'\d+', path.name)
    return int(match.group()) if match else path.name

try:
    checkpoint_paths = sorted(checkpoint_paths, key=get_checkpoint_step)
except Exception:
    checkpoint_paths = sorted(checkpoint_paths)

print(f"Found {len(checkpoint_paths)} checkpoints to evaluate in order:")
for p in checkpoint_paths:
    print(f" - {p.name}")

# Setup Stockfish pool once
cpus_per_gpu = os.getenv("SLURM_CPUS_PER_GPU")
if cpus_per_gpu is not None:
    n_jobs = int(cpus_per_gpu) - 2
else:
    n_jobs = os.cpu_count() - 2

stockfish_path = base_path / ".." / "Stockfish" / "src" / "stockfish"
engine_pool = queue.Queue()
for _ in range(n_jobs):
    engine = SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Threads": 1, "Hash": 32})
    engine_pool.put(engine)

# Dataset caching
dataset = None
def get_dataset(use_context, context_dataset):
    global dataset
    if dataset is not None:
        return dataset
    if use_context:
        dataset_name = "trainset.pt" if context_dataset == "train" else "testset.pt"
        dataset = torch.load(base_path / "dataset" / "with_best_move" / dataset_name, weights_only=False, map_location="cpu")
        return dataset
    return None

def process_puzzle(fen_tokens, move_tokens, base_theme, base_rating, device):
    entry = {
        "target_themes": base_theme,
        "target_rating": base_rating,
        "fen": None,
        "best_move": None,
        "is_legal": False,
        "is_puzzle": False,
        "counter_intuitive": None,
        "actual_themes": None,
        "themes_match": None,
        "main_line": None
    }

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
            
            if base_theme is not None:
                entry["themes_match"] = theme_reward(base_theme, existing_themes)

        return entry
    finally:
        engine_pool.put(engine)

try:
    for cp_path in checkpoint_paths:
        output_file_name = cp_path.with_suffix(".csv").name
        output_path = output_dir / output_file_name
        
        if output_path.exists():
            print(f"\nSkipping {cp_path.name} as output already exists at {output_path}", flush=True)
            continue
            
        print(f"\n=== Evaluating checkpoint: {cp_path.name} ===", flush=True)
        checkpoint = torch.load(cp_path, map_location="cpu", weights_only=False)
        
        config = checkpoint["config"]
        model = MaskedDiffusion(config)
        model.load_state_dict(checkpoint["model"])
        model.to(device=device)
        
        results_list = []
        n = args.n_fens
        batch_size = 5000
        
        total_batches = (n + batch_size - 1) // batch_size
        
        for iteration in range(total_batches):
            current_batch_size = min(batch_size, n - len(results_list))
            if current_batch_size <= 0:
                break
                
            print(f"Batch {iteration + 1}/{total_batches}", flush=True)
            
            ds = get_dataset(config.use_context, args.context_dataset)
            if config.use_context and ds is not None:
                indices = torch.randint(0, len(ds), (current_batch_size,)).tolist()
                sampled_items = [ds[idx] for idx in indices]
                
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
                tokens = module.sample(
                    themes_one_hot,
                    scaled_ratings,
                    batch_size=current_batch_size,
                    steps=args.steps,
                    temperature=args.temperature,
                    generate_move_last=args.generate_move_last
                )
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
                    ) for i in range(current_batch_size)
                ]
                batch_results = list(executor.map(lambda p: process_puzzle(*p), args_list))
            print("Processing time:", perf_counter() - start2, flush=True)
            print("Total time:", perf_counter() - start, flush=True)
            
            results_list.extend(batch_results)
            
        df = pd.DataFrame(results_list)
        df.to_csv(output_path, index=False)
        print(f"Saved results to {output_path}", flush=True)
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

finally:
    print("Shutting down Stockfish pool...", flush=True)
    while not engine_pool.empty():
        try:
            engine = engine_pool.get_nowait()
            engine.quit()
        except queue.Empty:
            break
    print("Done.", flush=True)
