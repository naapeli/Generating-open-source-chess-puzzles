import pandas as pd
import chess

from pathlib import Path


counter = 0
def get_puzzle_fen(row):
    global counter
    counter += 1
    if counter % 10000 == 0:
        print(f"{counter} rows")

    board = chess.Board(row["FEN"])
    board.push_uci(row["Moves"].split()[0])
    return board.fen()

base_path = Path("../dataset")
df = pd.read_csv(base_path / "lichess_db_puzzle.csv")
new_df = pd.DataFrame(df[["FEN", "Moves", "Rating", "Themes"]])
print(len(new_df))
new_df["Puzzle_FEN"] = df.apply(get_puzzle_fen, axis=1)
print(len(new_df))
new_df.to_csv(base_path / "dataset.csv")
