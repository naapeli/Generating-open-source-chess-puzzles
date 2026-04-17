import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import chess
import chess.engine
from pathlib import Path
import argparse
import queue
from joblib import Parallel, delayed
from tqdm import tqdm
import os
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from MaskedDiffusion.model import MaskedDiffusion
from Config import Config
from tokenization.tokenization import tokens_to_fen, tokens_to_move, scale_ratings, tokenize_fen
import pandas as pd
import numpy as np


def predict_moves(model, fen_tokens, themes, ratings, steps=512, temperature=1.0):
    batch_size = len(ratings)
    device = ratings.device
    config = model.config
    mask_token = config.mask_token
    
    move_tokens = torch.full((batch_size, config.move_length), mask_token, device=device, dtype=torch.long)
    tokens = torch.cat([fen_tokens, move_tokens], dim=1)
    
    seq_length = config.fen_length + config.move_length
    
    T_grid = torch.linspace(0, 1, steps + 1, device=device)
    
    for i in range(steps, 0, -1):
        t = T_grid[i]
        s = T_grid[i - 1]
        alpha_t = config.masking_schedule(t)
        alpha_s = config.masking_schedule(s)
        
        with torch.no_grad():
            logits = model(tokens, themes, ratings)
        
        probs = F.softmax(logits / temperature, dim=2)
        
        p_unmask = (alpha_s - alpha_t) / (1.0 - alpha_t + 1e-13)
        p_mask = (1.0 - alpha_s) / (1.0 - alpha_t + 1e-13)
        
        probs = torch.cat([probs * p_unmask, torch.full((batch_size, seq_length, 1), p_mask, device=device, dtype=probs.dtype)], dim=2)

        is_masked = (tokens == mask_token)
        
        log_probs = torch.log(probs + 1e-13)
        u = torch.rand_like(log_probs)
        gumbel_noise = -torch.log(-torch.log(u + 1e-13) + 1e-13)
        new_samples = torch.argmax(log_probs + gumbel_noise, dim=-1)
        
        tokens = torch.where(is_masked, new_samples, tokens)
        
    return tokens[:, config.fen_length:]

def process_batch_item(fen_tokens, model_move_tokens, theme, rating, engine_pool):
    engine = engine_pool.get()
    fen = tokens_to_fen(fen_tokens)
    model_move = tokens_to_move(model_move_tokens)

    board = chess.Board(fen)
    
    try:
        model_move = chess.Move.from_uci(model_move)
        is_legal = model_move in board.legal_moves
    except:
        is_legal = False
    
    if not is_legal:
        engine_pool.put(engine)
        return {"cp_loss": None, "is_legal": False, "is_match": False, "is_mate_opp": False, "is_mate_match": False}
    
    limit = chess.engine.Limit(depth=15, time=10, nodes=8_000_000)
    best_move_info = engine.analyse(board, limit=limit)
    player_to_move = board.turn
    best_move = best_move_info["pv"][0]
    best_score = best_move_info["score"].pov(player_to_move)
    
    board.push(model_move)
    model_info = engine.analyse(board, limit=limit)
    model_score = model_info["score"].pov(player_to_move)
    board.pop()
    
    is_mate_opp = best_score.is_mate()
    is_mate_match = False
    cp_loss = None

    if is_mate_opp:
        best_mate = best_score.mate()
        if model_score.is_mate():
            model_mate = model_score.mate()
            if (best_mate > 0 and model_mate > 0) or (best_mate < 0 and model_mate < 0):
                if abs(model_mate) <= abs(best_mate) - 1:
                    is_mate_match = True
    elif model_score.is_mate():
        pass 
    else:
        cp_loss = max(0, best_score.score() - model_score.score())
    
    engine_pool.put(engine)
    return {
        "cp_loss": cp_loss, 
        "is_legal": True, 
        "is_match": (model_move == best_move),
        "is_mate_opp": is_mate_opp,
        "is_mate_match": is_mate_match
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--stockfish_path", type=str, default="./Stockfish/src/stockfish")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_samples", type=int, default=None, help="Number of samples to evaluate (default: all)")
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--n_jobs", type=int, default=16)
    parser.add_argument("--puzzles", action="store_true")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ====================== LOAD MODEL ======================
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = MaskedDiffusion(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    
    # ====================== LOAD DATASET ======================
    if args.puzzles:
        testset_path = Path(".") / "src" / "dataset" / "with_best_move" / "testset.pt"
        testset = torch.load(testset_path, map_location="cpu", weights_only=False)
        
        if args.n_samples:
            indices = torch.arange(min(args.n_samples, len(testset)))
            from torch.utils.data import Subset
            testset = Subset(testset, indices)
    else:
        csv_path = Path(".") / "src" / "dataset" / "dataset.csv"
        df = pd.read_csv(csv_path, nrows=args.n_samples)
        
        fens = np.stack(df["FEN"].apply(tokenize_fen).values, dtype=np.int8)
        
        themes = np.zeros((len(df), config.n_themes), dtype=np.float32)
        scaled_rating = scale_ratings(2000)
        ratings = np.full((len(df),), scaled_rating, dtype=np.float32)
        
        dummy_moves = np.zeros((len(df), 5), dtype=np.int8)
        
        testset = []
        for i in range(len(df)):
            testset.append((
                torch.from_numpy(fens[i]),
                torch.from_numpy(dummy_moves[i]),
                torch.from_numpy(themes[i]),
                torch.tensor(ratings[i])
            ))
            
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    
    # ====================== ENGINE POOL ======================
    stockfish_path = Path(args.stockfish_path)
    
    engine_pool = queue.Queue()
    for _ in range(args.n_jobs):
        engine_pool.put(chess.engine.SimpleEngine.popen_uci(stockfish_path))
    
    # ====================== EVALUATION LOOP ======================
    total_cp_loss = 0
    total_legal = 0
    total_matches = 0
    count = 0

    total_cp_count = 0
    total_mate_opps = 0
    total_mate_matches = 0
    cp_losses = []

    for i, (fens, _, themes, ratings) in enumerate(testloader):
        print(f"Batch {i + 1}/{len(testloader)}", flush=True)
        fens = fens.to(device=device, dtype=torch.long)
        themes = themes.to(device=device, dtype=torch.float32)
        ratings = ratings.to(device=device, dtype=torch.float32)
        
        pred_moves = predict_moves(model, fens, themes, ratings, steps=args.steps)
        
        fens_cpu = fens.cpu()
        pred_moves_cpu = pred_moves.cpu()
        themes_cpu = themes.cpu()
        ratings_cpu = ratings.cpu()
        
        batch_results = Parallel(n_jobs=args.n_jobs, backend="threading")(delayed(process_batch_item)(fens_cpu[i], pred_moves_cpu[i], themes_cpu[i], ratings_cpu[i], engine_pool) for i in range(len(fens)))
        
        for res in batch_results:
            if res:
                if res["cp_loss"] is not None:
                    total_cp_loss += res["cp_loss"]
                    total_cp_count += 1
                    cp_losses.append(res["cp_loss"])
                
                total_legal += res["is_legal"]
                total_matches += res["is_match"]
                
                if res["is_mate_opp"]:
                    total_mate_opps += 1
                    if res["is_mate_match"]:
                        total_mate_matches += 1
                
                count += 1

    print("="*30)
    print(f"Average CP Loss:   {total_cp_loss/total_cp_count:.2f}")
    print(f"Legal Move Rate:   {total_legal/count:.2%}")
    print(f"Move Accuracy:     {total_matches/count:.2%}")
    print(f"Mate Accuracy:     {total_mate_matches/total_mate_opps:.2%}")
    print("="*30)

    while not engine_pool.empty():
        engine = engine_pool.get()
        engine.quit()

    # ====================== SAVE HISTOGRAM ======================
    if cp_losses:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(cp_losses, bins=50, color="steelblue", edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Centipawn Loss", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Distribution of Centipawn Losses", fontsize=14)
        ax.axvline(total_cp_loss / total_cp_count, color="tomato", linestyle="--", linewidth=1.5, label=f"Mean: {total_cp_loss / total_cp_count:.1f}")
        ax.legend()
        fig.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hist_path = Path(".") / f"cp_loss_distribution_{timestamp}.png"
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)

if __name__ == "__main__":
    main()
