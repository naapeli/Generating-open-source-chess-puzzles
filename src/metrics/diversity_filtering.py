import os
from pathlib import Path
import json

import numpy as np
import torch
# from torchaudio.functional import edit_distance
from rapidfuzz.distance import Levenshtein


def PV_distance(pv1: str, pv2: str) -> bool:
    pv1 = pv1.split(" ", 1)[0]  # only check if the first move is the same
    pv2 = pv2.split(" ", 1)[0]  # only check if the first move is the same
    # we do not divide by the max amount of moves in the pv, as they are both one.
    return Levenshtein.distance(pv1, pv2) >= 1  # max length of edit_distance is max(len(pv1), len(pv2))

def board_distance(fen1: str, fen2: str) -> bool:
    return Levenshtein.distance(fen1, fen2) >= 6


class ReplayBuffer:
    def __init__(self, capacity, path="./replay_buffer"):
        self.capacity = capacity
        self.path = Path(path)
        
        self.ptr = 0
        self.size = 0
        self.meta_path = self.path / "metadata.json"

        os.makedirs(self.path, exist_ok=True)
        if os.path.exists(self.meta_path):
            self.load_metadata()
            mode = "r+"
        else:
            mode = "w+"

        self.fen_max_length = 88
        self.pv_max_length = 100
        self.theme_max_length = 138  # dataset = pd.read_csv("./dataset/dataset.csv"); print(dataset["Themes"].apply(len).max())

        self.fens = np.memmap(self.path / "fens.npy", dtype="S" + str(self.fen_max_length), mode=mode, shape=(capacity,))
        self.pvs = np.memmap(self.path / "pvs.npy", dtype="S" + str(self.pv_max_length), mode=mode, shape=(capacity,))
        self.themes = np.memmap(self.path / "themes.npy", dtype="S" + str(self.theme_max_length), mode=mode, shape=(capacity,))
        self.ratings = np.memmap(self.path / "ratings.npy", dtype="float32", mode=mode, shape=(capacity,))

    def add(self, fen: str, pv: str, themes: list[str], rating: float):
        themes: str = " ".join(themes)
        if len(fen) > self.fen_max_length or len(pv) > self.pv_max_length or len(themes) > self.theme_max_length:
            return  # safe exit if the position or the solution is too long to be stored

        self.fens[self.ptr] = fen.encode("ascii")
        self.pvs[self.ptr] = pv.encode("ascii")
        self.themes[self.ptr] = themes.encode("ascii")
        self.ratings[self.ptr] = rating

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.save_metadata()

        if self.ptr % 1000 == 0:
            # save the file every 1000 positions
            self.fens.flush()
            self.pvs.flush()
            self.themes.flush()

    def sample(self, batch_size):
        if batch_size > self.size:
            raise RuntimeError()
        
        ind = np.random.choice(self.size, size=batch_size, replace=False)

        fens = [s.decode("utf-8").strip() for s in self.fens[ind]]
        pvs = [s.decode("utf-8").strip() for s in self.pvs[ind]]
        themes = [s.decode("utf-8").strip().split(" ") for s in self.themes[ind]]
        ratings = ratings = torch.from_numpy(self.ratings[ind]).clone()

        return fens, pvs, themes, ratings

    def save_metadata(self):
        with open(self.meta_path, "w") as f:
            json.dump({"ptr": self.ptr, "size": self.size}, f)

    def load_metadata(self):
        with open(self.meta_path, "r") as f:
            data = json.load(f)
            self.ptr = data["ptr"]
            self.size = data["size"]


if __name__ == "__main__":
    capacity = 20
    buffer = ReplayBuffer(capacity)

    buffer.add("q3k1nr/1pp1nQpp/3p4/1P2p3/4P3/B1PP1b2/B5PP/5K2 b k - 0 17", "e8d7 a2e6 d7d8 f7f8", ["mate", "mateIn2", "middlegame", "short"], 1760)
    buffer.add("r3r1k1/p4ppp/2p2n2/1p6/3P1qb1/2NQR3/PPB2PP1/R1B3K1 w - - 5 18", "e3g3 e8e1 g1h2 e1c1 a1c1 f4h6 h2g1 h6c1", ["advantage", "attraction", "fork", "middlegame", "sacrifice", "veryLong"], float(np.float64(2671)))
    buffer.add("q3k1nr/1pp1nQpp/3p4/1P2p3/4P3/B1PP1b2/B5PP/5K2 b k - 0 17", "e8d7 a2e6 d7d8 f7f8", ["advantage", "fork", "long"], 2235)
    buffer.add("r3r1k1/p4ppp/2p2n2/1p6/3P1qb1/2NQR3/PPB2PP1/R1B3K1 w - - 5 18", "e3g3 e8e1 g1h2 e1c1 a1c1 f4h6 h2g1 h6c1", ["advantage", "discoveredAttack", "master", "middlegame", "short"], 998)
    print(buffer.fens)
    print(buffer.pvs)
    print(buffer.themes)
    print(buffer.ratings)

    del buffer

    buffer = ReplayBuffer(capacity)
    print(buffer.fens)
    print(buffer.pvs)
    print(buffer.themes)
    print(buffer.ratings)

    sampled_fens, sampled_pvs, sampled_themes, sampled_ratings = buffer.sample(2)
    print(sampled_fens)
    print(sampled_pvs)
    print(sampled_themes)
    print(sampled_ratings)
