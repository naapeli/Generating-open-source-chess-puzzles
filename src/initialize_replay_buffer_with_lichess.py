import pandas as pd
from pathlib import Path
import os
from joblib import Parallel, delayed

from chess.engine import SimpleEngine

from metrics.themes import get_unique_puzzle_from_fen, counter_intuitive
from metrics.cook import cook
from metrics.diversity_filtering import ReplayBuffer

base_path = Path("./src")
dataset_path = base_path / "dataset" / "dataset.csv"
stockfish_path = base_path / ".." / "Stockfish" / "src" / "stockfish"
replay_buffer_path = base_path / "dataset" / "rl"

num_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 2))  # os.cpu_count() is wrong if one requests less cores
chunksize = 1000

capacity = 200_000
buffer = ReplayBuffer(capacity, replay_buffer_path)

def process_batch(chunk):
    engine = SimpleEngine.popen_uci(stockfish_path, timeout=60)
    engine.configure({"Hash": 16384 // num_cores})
    
    results = []
    for row in chunk:
        engine.configure({"Clear Hash": None})
        puzzle = get_unique_puzzle_from_fen(row.Puzzle_FEN, engine)
        if puzzle is None:
            continue
        engine.configure({"Clear Hash": None})
        if not counter_intuitive(row.Puzzle_FEN, engine):
            continue

        themes = cook(puzzle, engine)
        if len(set(themes).intersection(row.Themes.split(" "))) > len(themes) / 2:
            results.append((row.Puzzle_FEN, row.Moves))
    
    engine.quit()
    return results

dataset = pd.read_csv(dataset_path, chunksize=chunksize)
n_positions_added = 0
for chunk in dataset:
    chunks = [chunk[i::num_cores].itertuples(index=False) for i in range(num_cores)]
    parallel_results = Parallel(n_jobs=num_cores)(delayed(process_batch)(parallel_chunk) for parallel_chunk in chunks)
    for batch in parallel_results:
        for fen, moves in batch:
            n_positions_added += 1
            buffer.add(fen, moves)
    print(n_positions_added, flush=True)
    
    if n_positions_added >= capacity // 2:
        break
