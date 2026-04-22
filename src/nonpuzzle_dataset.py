import os
import chess
import chess.pgn
import queue
import torch
import numpy as np
from joblib import Parallel, delayed
from chess.engine import SimpleEngine, Limit
from torch.utils.data import TensorDataset

from pathlib import Path

from tokenization.tokenization import tokenize_fen, tokenize_move


base_path = Path("./src")
stockfish_path = "./Stockfish/src/stockfish"
n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

engine_pool = queue.Queue()
for _ in range(n_jobs):
    engine_pool.put(SimpleEngine.popen_uci(stockfish_path))

checkpoint_path = base_path / "dataset" / "normal_positions" / "checkpoint.pt"
checkpoint_interval = 50

def load_checkpoint():
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}...", flush=True)
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        return checkpoint["dataset"], checkpoint["game_count"]
    return [], 0

dataset, last_game_idx = load_checkpoint()

def process_fen(fen):
    engine = engine_pool.get()
    try:
        board = chess.Board(fen)
        result = engine.play(board, Limit(time=0.2))

        tokenized_fen = tokenize_fen(fen)
        tokenized_move = tokenize_move(result.move.uci())
        return tokenized_fen, tokenized_move
    except Exception:
        print("Error")
        return None
    finally:
        engine_pool.put(engine)

batch_size = 16_384
fens_batch = []
batches_since_checkpoint = 0

game_count = 0

with open(base_path / "dataset" / "normal_positions" / "large_position_dataset.pgn") as f:
    while True:
        game = chess.pgn.read_game(f)
        if game is None:
            break
        
        game_count += 1
        if game_count <= last_game_idx:
            if game_count % 10000 == 0:
                print(f"Skipping games {game_count}/{last_game_idx}...", flush=True)
            continue

        if game_count % 10000 == 0:
            print(f"Read {game_count} games, currently {len(dataset)} FENs in the dataset", flush=True)
            
        board = game.board()
        for i, move in enumerate(game.mainline_moves()):
            board.push(move)
            if i > 5 and i % 3 == 0 and not board.is_game_over():
                fens_batch.append(board.fen())
                
                if len(fens_batch) >= batch_size:
                    batch_results = Parallel(n_jobs=n_jobs, backend="threading")(delayed(process_fen)(fen) for fen in fens_batch)
                    dataset.extend([r for r in batch_results if r is not None])
                    fens_batch = []
                    batches_since_checkpoint += 1
                    
                    if batches_since_checkpoint >= checkpoint_interval:
                        torch.save({"dataset": dataset, "game_count": game_count}, checkpoint_path)
                        print(f"Checkpoint saved at game {game_count}, dataset size {len(dataset)}", flush=True)
                        batches_since_checkpoint = 0

if fens_batch:
    batch_results = Parallel(n_jobs=n_jobs, backend="threading")(delayed(process_fen)(fen) for fen in fens_batch)
    dataset.extend([r for r in batch_results if r is not None])

while not engine_pool.empty():
    engine = engine_pool.get()
    engine.quit()

print(f"Finished processing. Total positions: {len(dataset)}.", flush=True)

tokenized_fens = torch.tensor([d[0] for d in dataset], dtype=torch.int8)
tokenized_moves = torch.tensor([d[1] for d in dataset], dtype=torch.int8)

tensor_dataset = TensorDataset(tokenized_fens, tokenized_moves)
output_path = base_path / "dataset" / "normal_positions" / "large_position_dataset.pt"
torch.save(tensor_dataset, output_path)

if checkpoint_path.exists():
    checkpoint_path.unlink()
    print("Checkpoint removed.", flush=True)
