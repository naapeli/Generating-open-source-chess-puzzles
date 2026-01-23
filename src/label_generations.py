from pathlib import Path
import os
import pandas as pd
from chess.engine import SimpleEngine

from metrics.themes import legal, get_unique_puzzle_from_fen, counter_intuitive
from metrics.cook import cook


base_path = Path("./src")

df = pd.read_csv(base_path / "generated_fens.txt", header=None, names=["index", "FEN", "rating", "theme", "legal"])

stockfish_path = "./Stockfish/src/stockfish"
with SimpleEngine.popen_uci(stockfish_path) as engine:
    engine.configure({"Threads": os.cpu_count()})
    engine.configure({"Hash": 16384})

    uniques = []
    themes = []
    counter_intuitives = []

    for i, row in df.iterrows():
        print("Position", i, flush=True)
        fen = row["FEN"]
        is_unique = None
        puzzle_themes = []
        is_counter_intuitive = None

        if legal(fen):
            engine.configure({"Clear Hash": None})
            puzzle = get_unique_puzzle_from_fen(fen, engine)
            is_unique = puzzle is not None
            if is_unique:
                print(puzzle.game)
                puzzle_themes = cook(puzzle, engine)
                print("Found themes:", " ".join(puzzle_themes))
                print("Base theme: ", row["theme"])
            is_counter_intuitive = counter_intuitive(fen, engine)

        uniques.append(is_unique)
        counter_intuitives.append(is_counter_intuitive)
        themes.append(" ".join(puzzle_themes))

    df["unique"] = uniques
    df["counter_intuitive"] = counter_intuitives
    df["themes"] = themes

    df.to_csv(base_path / "processed_puzzles.csv", index=False)
