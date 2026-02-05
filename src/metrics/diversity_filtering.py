import os
from pathlib import Path
import json

import numpy as np
from torchaudio.functional import edit_distance


def PV_distance(pv1: str, pv2: str) -> bool:
    return edit_distance(pv1, pv2) / max(len(pv1), len(pv2)) >= 1

def board_distance(fen1: str, fen2: str) -> bool:
    return edit_distance(fen1, fen2) >= 6


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

        self.fens = np.memmap(self.path / "fens.npy", dtype="S" + str(self.fen_max_length), mode=mode, shape=(capacity,))
        self.pvs = np.memmap(self.path / "pvs.npy", dtype="S" + str(self.pv_max_length), mode=mode, shape=(capacity,))

    def add(self, fen: str, pv: str):
        if len(fen) > self.fen_max_length or len(pv) > self.pv_max_length:
            return  # safe exit if the position or the solution is too long to be stored

        self.fens[self.ptr] = fen.encode("ascii")
        self.pvs[self.ptr] = pv.encode("ascii")

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.save_metadata()

        if self.ptr % 1000 == 0:
            # save the file every 1000 positions
            self.fens.flush()
            self.pvs.flush()

    def sample(self, batch_size):
        if batch_size > self.size:
            raise RuntimeError()
        
        ind = np.random.choice(self.size, size=batch_size, replace=False)

        fens = [s.decode("utf-8").strip() for s in self.fens[ind]]
        pvs = [s.decode("utf-8").strip() for s in self.pvs[ind]]

        return fens, pvs

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

    buffer.add("q3k1nr/1pp1nQpp/3p4/1P2p3/4P3/B1PP1b2/B5PP/5K2 b k - 0 17", "e8d7 a2e6 d7d8 f7f8")
    buffer.add("r3r1k1/p4ppp/2p2n2/1p6/3P1qb1/2NQR3/PPB2PP1/R1B3K1 w - - 5 18", "e3g3 e8e1 g1h2 e1c1 a1c1 f4h6 h2g1 h6c1")
    buffer.add("q3k1nr/1pp1nQpp/3p4/1P2p3/4P3/B1PP1b2/B5PP/5K2 b k - 0 17", "e8d7 a2e6 d7d8 f7f8")
    buffer.add("r3r1k1/p4ppp/2p2n2/1p6/3P1qb1/2NQR3/PPB2PP1/R1B3K1 w - - 5 18", "e3g3 e8e1 g1h2 e1c1 a1c1 f4h6 h2g1 h6c1")
    print(buffer.fens)
    print(buffer.pvs)

    del buffer

    buffer = ReplayBuffer(capacity)
    print(buffer.fens)
    print(buffer.pvs)

    sampled_fens, sampled_pvs = buffer.sample(2)
    print(sampled_fens)
    print(sampled_pvs)
