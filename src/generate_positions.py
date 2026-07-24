import argparse
import queue
import torch
from pathlib import Path
import pandas as pd
from chess.engine import SimpleEngine
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
import os
import random

from MaskedDiffusion.model import MaskedDiffusion
from rl.espo import generate_random_themes, theme_reward
from tokenization.tokenization import theme_preprocessor, scale_ratings, tokens_to_fen, tokens_to_move, unscale_ratings
from metrics.themes import legal, get_unique_puzzle_from_fen, counter_intuitive
from metrics.cook import cook
from MaskingSchedule.MaskingSchedule import string_to_schedule
from torch.distributed import init_process_group, destroy_process_group, barrier


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
parser.add_argument("--batch_size", type=int, default=1024)
args = parser.parse_args()

distributed = False
rank = 0
local_rank = 0
world_size = 1
master_process = True

if "SLURM_NTASKS" in os.environ:
    world_size = int(os.environ["SLURM_NTASKS"])
    distributed = world_size > 1
    if distributed:
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        master_process = (rank == 0)

if distributed:
    assert torch.cuda.is_available(), "CUDA must be available for distributed generation"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mix rank into the seed to ensure different processes generate different puzzles
initial_seed = torch.seed()
rank_seed = (initial_seed + rank) % (2**32)
torch.manual_seed(rank_seed)
random.seed(rank_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(rank_seed)

base_path = Path("./src")
checkpoint = torch.load(base_path / "runs" / args.run_type / args.run_name / args.checkpoint_name, map_location="cpu", weights_only=False)

config = checkpoint["config"]
model = MaskedDiffusion(config)
model.load_state_dict(checkpoint["model"])
model.to(device=device)
model.eval()

n = args.n_fens
batch_size = args.batch_size


slurm_cpus = os.getenv("SLURM_CPUS_PER_GPU")
if slurm_cpus is not None:
    n_jobs = int(slurm_cpus) - 2
else:
    n_jobs = max(1, os.cpu_count() - 2)
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
    entry = {"target_themes": base_theme, "target_rating": base_rating, "fen": None, "best_move": None, "is_legal": False, "is_puzzle": False, "counter_intuitive": None, "counter_intuitive_value": None, "actual_themes": None, "themes_match": None, "main_line": None}

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
        entry["counter_intuitive"], entry["counter_intuitive_value"] = counter_intuitive(fen, engine, return_value=True)
        puzzle = get_unique_puzzle_from_fen(fen, engine)
        
        if puzzle is not None:
            entry["is_puzzle"] = True
            entry["main_line"] = " ".join([move.uci() for move in puzzle.mainline])
            existing_themes = cook(puzzle, engine)
            entry["actual_themes"] = existing_themes
            
            if config.use_context:
                entry["themes_match"] = theme_reward(base_theme, existing_themes)

        return entry

    finally:
        engine_pool.put(engine)


# Distribute total n_fens among SLURM tasks
local_n = args.n_fens // world_size
if rank < args.n_fens % world_size:
    local_n += 1

output_path = base_path / "Generate_positions" / args.output_file
output_path.parent.mkdir(parents=True, exist_ok=True)

if distributed:
    local_output_path = output_path.with_name(f"{output_path.stem}_rank{rank}{output_path.suffix}")
else:
    local_output_path = output_path

# Ensure the local output file is cleared at the start of generation
if local_output_path.exists():
    local_output_path.unlink()

remaining = local_n
iteration = 0
total_iterations = (local_n + batch_size - 1) // batch_size

# Re-use ThreadPoolExecutor across all batches for max performance
executor = ThreadPoolExecutor(max_workers=n_jobs)
try:
    while remaining > 0:
        current_batch_size = min(batch_size, remaining)
        iteration += 1
        if master_process:
            print(f"Iteration {iteration} / {total_iterations} (generating batch of size {current_batch_size})", flush=True)
        
        if config.use_context:
            indices = torch.randint(0, len(dataset), (current_batch_size,)).tolist()
            sampled_items = [dataset[idx] for idx in indices]
            
            sampled_themes = torch.stack([torch.as_tensor(item[2]) for item in sampled_items])
            sampled_ratings = torch.stack([torch.as_tensor(item[3]) for item in sampled_items])
            
            base_themes = theme_preprocessor.inverse_transform(sampled_themes.numpy())
            base_ratings = unscale_ratings(sampled_ratings).tolist()
            
            themes_one_hot = sampled_themes.to(device=device, dtype=torch.float32)
            scaled_ratings = sampled_ratings.to(device=device, dtype=torch.float32)
        else:
            themes_one_hot = None
            scaled_ratings = None
            base_themes = None
            base_ratings = None

        module = model.module if hasattr(model, "module") else model
        start = perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            tokens = module.sample(themes_one_hot, scaled_ratings, batch_size=current_batch_size, steps=args.steps, temperature=args.temperature, generate_move_last=args.generate_move_last)
        
        # Copy to CPU once to avoid thread-level GPU synchronization bottleneck
        tokens_cpu = tokens.cpu()
        print(f"[Rank {rank}] Sampling time: {perf_counter() - start:.4f}s", flush=True)
        
        if config.predict_moves:
            fen_tokens = tokens_cpu[:, :config.fen_length]
            move_tokens = tokens_cpu[:, config.fen_length:]
        else:
            fen_tokens = tokens_cpu
            move_tokens = None

        start2 = perf_counter()
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
        print(f"[Rank {rank}] Processing time: {perf_counter() - start2:.4f}s", flush=True)
        print(f"[Rank {rank}] Total iteration time: {perf_counter() - start:.4f}s", flush=True)
        
        # Write batch incrementally to file
        df_batch = pd.DataFrame(batch_results)
        write_header = not local_output_path.exists()
        df_batch.to_csv(local_output_path, mode='a', index=False, header=write_header)
        
        remaining -= current_batch_size
finally:
    executor.shutdown(wait=True)

while not engine_pool.empty():
    engine = engine_pool.get()
    engine.quit()

if distributed:
    # Synchronize all ranks to ensure writing is complete
    barrier()
    
    # Merge rank files on Rank 0
    if master_process:
        dfs = []
        for r in range(world_size):
            r_path = output_path.with_name(f"{output_path.stem}_rank{r}{output_path.suffix}")
            if r_path.exists():
                try:
                    dfs.append(pd.read_csv(r_path))
                    r_path.unlink()
                except Exception as e:
                    print(f"Error reading/deleting rank file {r_path}: {e}", flush=True)
        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_df.to_csv(output_path, index=False)
            print(f"Successfully merged {len(merged_df)} positions and saved to {output_path}", flush=True)
        else:
            print(f"Warning: No rank files found to merge.", flush=True)
            
    destroy_process_group()
else:
    print(f"Successfully generated and saved {local_n} positions to {output_path}", flush=True)
