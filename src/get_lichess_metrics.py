from pathlib import Path

import pandas as pd
from chess.engine import SimpleEngine

from metrics.themes import get_unique_puzzle_from_fen, counter_intuitive


base_path = Path(".")
dataset_path = base_path / "src" / "dataset" / "dataset.csv"
stockfish_path = base_path / "Stockfish" / "src" / "stockfish"

n = 1000
n_unique_lichess = 0
n_counter_intuitive_lichess = 0
with SimpleEngine.popen_uci(stockfish_path) as engine:
    for chunk in pd.read_csv(dataset_path, chunksize=n):
        for i, fen in enumerate(chunk["Puzzle_FEN"]):
            unique = get_unique_puzzle_from_fen(fen, engine) is not None
            counter_intuitive_position = counter_intuitive(fen, engine)
            n_unique_lichess += unique
            n_counter_intuitive_lichess += unique and counter_intuitive_position  # sum over counter intuitive positions which are unique
            if i % 10 == 0: print(i, fen, flush=True)
        break

print(n_unique_lichess / n, n_counter_intuitive_lichess / n_unique_lichess, n_counter_intuitive_lichess / n)
