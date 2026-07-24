import os
from pathlib import Path
import json
import re

import numpy as np
import torch
# from torchaudio.functional import edit_distance
from rapidfuzz.distance import Levenshtein
import chess


def fen_to_padded(fen):  # make the format the same as 
    board, side, castling, enpassant = fen.split(" ")[:4]
    board = re.sub(r"\d", lambda digit: "." * int(digit.group()), board)
    board = re.sub("/", "", board)
    return "".join([side, board])
    # return " ".join([board, side])
    # castling = "".join([char if char in castling else "." for char in "KQkq"])
    # enpassant = ".." if enpassant == "-" else enpassant
    # return " ".join([board, side, castling, enpassant])

def PV_distance(pv1: str, pv2: str) -> bool:
    return get_pv_distance(pv1, pv2) >= 1  # max length of edit_distance is max(len(pv1), len(pv2))

def board_distance(fen1: str, fen2: str) -> bool:
    return get_board_distance(fen1, fen2) >= 6

def get_pv_distance(pv1: str, pv2: str) -> int:
    if not pv1 or not pv2:
        return 0
    pv1 = pv1.split(" ", 1)[0]  # only check if the first move is the same
    pv2 = pv2.split(" ", 1)[0]  # only check if the first move is the same
    return Levenshtein.distance(pv1, pv2)

def get_board_distance(fen1: str, fen2: str) -> int:
    return Levenshtein.distance(fen_to_padded(fen1), fen_to_padded(fen2))

def get_opponent_pv_distance(pv1: str, pv2: str) -> int:
    opponent_pv1 = pv1.split(" ")
    opponent_pv2 = pv2.split(" ")
    return Levenshtein.distance(opponent_pv1[1], opponent_pv2[1]) if len(opponent_pv1) >= 2 and len(opponent_pv2) >= 2 else 5

def get_abstracted_pv(fen: str, pv: str) -> list:
    if not fen or not pv:
        return []
    try:
        board = chess.Board(fen)
    except Exception:
        return []

    abstracted = []
    for move_str in pv.split(" "):
        if not move_str:
            continue
        try:
            move = chess.Move.from_uci(move_str)
        except ValueError:
            break
        if not board.is_legal(move):
            break

        piece = board.piece_at(move.from_square)
        if piece is None:
            break
        piece_type = piece.piece_type

        # Determine direction
        from_file = chess.square_file(move.from_square)
        from_rank = chess.square_rank(move.from_square)
        to_file = chess.square_file(move.to_square)
        to_rank = chess.square_rank(move.to_square)
        dx = to_file - from_file
        dy = to_rank - from_rank

        if dx == 0:
            direction = "V"
        elif dy == 0:
            direction = "H"
        elif abs(dx) == abs(dy):
            direction = "D"
        elif (abs(dx) == 1 and abs(dy) == 2) or (abs(dx) == 2 and abs(dy) == 1):
            direction = "N"
        else:
            direction = "O"

        # Apply move to check state
        board.push(move)
        is_check = board.is_check()
        is_mate = board.is_checkmate()

        abstracted.append((piece_type, direction, is_check, is_mate))

    return abstracted


def abstract_moves_equal(move1: tuple, move2: tuple) -> bool:
    piece1, dir1, ch1, mate1 = move1
    piece2, dir2, ch2, mate2 = move2

    # Basic flags must match
    # if ch1 != ch2 or mate1 != mate2 or dir1 != dir2:
    if ch1 != ch2 or mate1 != mate2:
        return False

    # Same piece type matches directly
    if piece1 == piece2:
        return True

    # Queen and Rook are equivalent for straight moves
    if dir1 == dir2 and dir1 in ("V", "H"):
        if {piece1, piece2} <= {chess.ROOK, chess.QUEEN}:
            return True

    # Queen and Bishop are equivalent for diagonal moves
    if dir1 == dir2 and dir1 == "D":
        if {piece1, piece2} <= {chess.BISHOP, chess.QUEEN}:
            return True

    return False


def get_abstracted_pv_hamming_distance(apv1: list, apv2: list) -> int:
    if len(apv1) != len(apv2) or len(apv1) == 0 or len(apv2) == 0:
        return max(len(apv1), len(apv2), 1)

    dist = 0
    for m1, m2 in zip(apv1, apv2):
        if not abstract_moves_equal(m1, m2):
            dist += 1
    return dist



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
