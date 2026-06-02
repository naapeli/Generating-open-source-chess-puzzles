import argparse
import time
import re
import pandas as pd
import numpy as np
from pathlib import Path
from rapidfuzz.process import cdist
from rapidfuzz.distance import Levenshtein

from metrics.diversity_filtering import fen_to_padded


def extract_pv_moves(pv_str, is_lichess=False):
    """
    Extracts moves from the PV string.
    For Lichess: 'Moves' contains opponent's move first, then the solution. Skip index 0.
    For generated: 'main_line' contains only the solution. Keep all.
    """
    if not isinstance(pv_str, str) or not pv_str.strip():
        return []
    moves = pv_str.strip().split()
    if is_lichess:
        return moves[1:]
    return moves


def compute_dataset_distances(fens_A, fens_B, pvs_A, pvs_B, chunk_size=5000, is_self=False):
    """
    Computes minimum board distances and the PV distances to those same board-wise nearest neighbors.
    """
    n_A = len(fens_A)
    n_B = len(fens_B)
    
    board_dists = np.empty(n_A, dtype=np.float32)
    pv_dists = np.empty(n_A, dtype=np.float32)

    # Pad FEN strings
    padded_A = [fen_to_padded(f) for f in fens_A]
    padded_B = [fen_to_padded(f) for f in fens_B]

    # Truncate PVs to length 6 as done in the paper
    truncated_A = [pv[:6] for pv in pvs_A]
    truncated_B = [pv[:6] for pv in pvs_B]

    for start_idx in range(0, n_A, chunk_size):
        end_idx = min(start_idx + chunk_size, n_A)
        chunk_A = padded_A[start_idx:end_idx]

        # Compute pairwise board Levenshtein distance matrix for the chunk
        board_dist_matrix = cdist(chunk_A, padded_B, scorer=Levenshtein.distance, workers=-1).astype(np.float32)

        if is_self:
            # For self-distance, set the diagonal element to a large value
            for i in range(len(chunk_A)):
                board_dist_matrix[i, start_idx + i] = 999999.0

        # Find the index of the board-wise nearest neighbor in dataset B
        nn_indices = np.argmin(board_dist_matrix, axis=1)
        board_dists[start_idx:end_idx] = np.min(board_dist_matrix, axis=1)

        # Compute the normalized PV distance to that board-wise nearest neighbor
        chunk_pv_dists = []
        for i in range(len(chunk_A)):
            j = nn_indices[i]
            pv_A = truncated_A[start_idx + i]
            pv_B = truncated_B[j]
            raw_pv_dist = Levenshtein.distance(pv_A, pv_B)
            max_len = max(len(pv_A), len(pv_B))
            chunk_pv_dists.append(raw_pv_dist / max(max_len, 1.0))
            
        pv_dists[start_idx:end_idx] = chunk_pv_dists

    return board_dists, pv_dists



def main():
    parser = argparse.ArgumentParser(description="Compute dataset distances following paper Table 3.")
    parser.add_argument("--generated_csv", type=str, default=None, help="Path to generated CSV file.")
    parser.add_argument("--lichess_csv", type=str, default=None, help="Path to Lichess CSV file.")
    parser.add_argument("--sample_size", type=int, default=10000, help="Sample size to align and evaluate (default 10,000). Use -1 to evaluate all.")
    parser.add_argument("--chunk_size", type=int, default=5000, help="Chunk size for evaluation (default 5000).")
    parser.add_argument("--no_filter", action="store_true", help="Do not filter by is_puzzle or validity.")
    args = parser.parse_args()

    if not args.generated_csv and not args.lichess_csv:
        print("Error: Please provide at least one of --generated_csv or --lichess_csv.", flush=True)
        return

    # 1. Load generated dataset
    df_gen = None
    if args.generated_csv:
        path = Path(args.generated_csv)
        if not path.exists():
            print(f"Error: {args.generated_csv} does not exist.", flush=True)
            return
        print(f"Loading generated dataset from {path}...", flush=True)
        df_gen = pd.read_csv(path)
        print(f"Loaded {len(df_gen)} rows.", flush=True)

        # Filter by puzzle flag
        if not args.no_filter and "is_puzzle" in df_gen.columns:
            df_gen = df_gen[df_gen["is_puzzle"] == True]
            print(f"Filtered to {len(df_gen)} puzzles (is_puzzle == True).", flush=True)

    # 2. Load Lichess dataset
    df_lic = None
    if args.lichess_csv:
        path = Path(args.lichess_csv)
        if not path.exists():
            print(f"Error: {args.lichess_csv} does not exist.", flush=True)
            return
        print(f"Loading Lichess dataset from {path}...", flush=True)
        # Since dataset.csv is huge (1GB), load it efficiently
        # If we need a sample, we can load a chunk or the first N rows
        if args.sample_size > 0:
            # Load a bit more to ensure we have enough puzzles after filtering if filtering is on
            nrows_to_load = args.sample_size * 5 if not args.no_filter else args.sample_size
            df_lic = pd.read_csv(path, nrows=nrows_to_load)
        else:
            df_lic = pd.read_csv(path)
        print(f"Loaded {len(df_lic)} rows from Lichess.", flush=True)

        if not args.no_filter:
            # In dataset.csv, everything is a puzzle, but we check if we need to clean or filter
            pass

    # 3. Sample and align sizes
    gen_fens, gen_pvs = [], []
    lic_fens, lic_pvs = [], []

    if df_gen is not None:
        # Get first N generated positions
        n_gen = len(df_gen)
        sample_n = args.sample_size if (args.sample_size > 0 and args.sample_size < n_gen) else n_gen
        df_gen_sampled = df_gen.head(sample_n) if sample_n < n_gen else df_gen
        
        gen_fens = df_gen_sampled["fen"].tolist()
        gen_pvs = [extract_pv_moves(pv, is_lichess=False) for pv in df_gen_sampled["main_line"]]
        print(f"Prepared {len(gen_fens)} generated samples.", flush=True)

    if df_lic is not None:
        # Get first N Lichess puzzles
        # Map FEN column from 'Puzzle_FEN' or 'FEN'
        fen_col = "Puzzle_FEN"
        pv_col = "Moves"
        
        # Filter rows that have non-null FEN and Moves
        df_lic_valid = df_lic.dropna(subset=[fen_col, pv_col])
        n_lic = len(df_lic_valid)
        sample_n = args.sample_size if (args.sample_size > 0 and args.sample_size < n_lic) else n_lic
        df_lic_sampled = df_lic_valid.head(sample_n) if sample_n < n_lic else df_lic_valid
        
        lic_fens = df_lic_sampled[fen_col].tolist()
        lic_pvs = [extract_pv_moves(pv, is_lichess=True) for pv in df_lic_sampled[pv_col]]
        print(f"Prepared {len(lic_fens)} Lichess samples.", flush=True)

    print("\n" + "="*50, flush=True)
    print("STARTING DISTANCE COMPUTATIONS", flush=True)
    print("="*50, flush=True)

    # 4. Compute Self-Distance for Generated Dataset
    if len(gen_fens) > 1:
        t0 = time.time()
        print(f"Computing generated dataset self-distances (sample size={len(gen_fens)})...", flush=True)
        gen_self_board, gen_self_pv = compute_dataset_distances(gen_fens, gen_fens, gen_pvs, gen_pvs, chunk_size=args.chunk_size, is_self=True)
        print(f"Done in {time.time() - t0:.2f}s.", flush=True)
        print(f"  -> Generated Self-Distance (Board): {np.mean(gen_self_board):.4f}", flush=True)
        print(f"  -> Generated Self-Distance (PV):    {np.mean(gen_self_pv):.4f}", flush=True)
        print("-"*50, flush=True)

    # 5. Compute Self-Distance for Lichess Dataset
    if len(lic_fens) > 1:
        t0 = time.time()
        print(f"Computing Lichess dataset self-distances (sample size={len(lic_fens)})...", flush=True)
        lic_self_board, lic_self_pv = compute_dataset_distances(lic_fens, lic_fens, lic_pvs, lic_pvs, chunk_size=args.chunk_size, is_self=True)
        print(f"Done in {time.time() - t0:.2f}s.", flush=True)
        print(f"  -> Lichess Self-Distance (Board): {np.mean(lic_self_board):.4f}", flush=True)
        print(f"  -> Lichess Self-Distance (PV):    {np.mean(lic_self_pv):.4f}", flush=True)
        print("-"*50, flush=True)

    # 6. Compute Cross-Distance (Novelty: Generated w.r.t Lichess)
    if len(gen_fens) > 0 and len(lic_fens) > 0:
        t0 = time.time()
        print(f"Computing Novelty distances (Generated w.r.t Lichess, gen_size={len(gen_fens)}, lic_size={len(lic_fens)})...", flush=True)
        novelty_board, novelty_pv = compute_dataset_distances(gen_fens, lic_fens, gen_pvs, lic_pvs, chunk_size=args.chunk_size, is_self=False)
        print(f"Done in {time.time() - t0:.2f}s.", flush=True)
        print(f"  -> Novelty/Lichess-Distance (Board): {np.mean(novelty_board):.4f}", flush=True)
        print(f"  -> Novelty/Lichess-Distance (PV):    {np.mean(novelty_pv):.4f}", flush=True)
        print("-"*50, flush=True)


if __name__ == "__main__":
    main()
