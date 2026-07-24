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


def compute_mean_and_se(data):
    """
    Computes the mean and standard error of the mean for a list/array of values.
    """
    if len(data) == 0:
        return None, None
    mean = np.mean(data)
    if len(data) <= 1:
        return mean, 0.0
    se = np.std(data, ddof=1) / np.sqrt(len(data))
    return mean, se


def compute_dataset_distances(fens_A, fens_B, pvs_A, pvs_B, chunk_size=5000, is_self=False):
    """
    Computes minimum board distances and the PV distances to those same board-wise nearest neighbors.
    """
    n_A = len(fens_A)
    n_B = len(fens_B)
    print("Dataset lengths:", n_A, n_B)
    
    board_dists = np.empty(n_A, dtype=np.float32)
    pv_dists = np.empty(n_A, dtype=np.float32)

    padded_A = [fen_to_padded(f) for f in fens_A]
    padded_B = [fen_to_padded(f) for f in fens_B]

    truncated_A = [pv[:6] for pv in pvs_A]
    truncated_B = [pv[:6] for pv in pvs_B]

    for start_idx in range(0, n_A, chunk_size):
        end_idx = min(start_idx + chunk_size, n_A)
        chunk_A = padded_A[start_idx:end_idx]

        print(f"   [+] Processing chunk {start_idx} to {end_idx}...", flush=True)

        board_dist_matrix = cdist(chunk_A, padded_B, scorer=Levenshtein.distance, workers=-1, score_hint=12, score_cutoff=65).astype(np.float32)

        if is_self:
            for i in range(len(chunk_A)):
                board_dist_matrix[i, start_idx + i] = 999999.0

        nn_indices = np.argmin(board_dist_matrix, axis=1)
        board_dists[start_idx:end_idx] = np.min(board_dist_matrix, axis=1)

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



def natural_sort_key(path):
    """
    Key for natural sorting of paths (e.g. checkpoint_1000.csv before checkpoint_2000.csv).
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', path.name)]


def main():
    parser = argparse.ArgumentParser(description="Compute dataset distances following paper Table 3.")
    parser.add_argument("--generated_csv", type=str, default=None, help="Path to generated CSV file or a directory containing them.")
    parser.add_argument("--lichess_csv", type=str, default=None, help="Path to Lichess CSV file.")
    parser.add_argument("--self_sample_size", type=int, default=10000, help="Sample size for generated self-distance (default 10,000). Use -1 to evaluate all.")
    parser.add_argument("--lichess_sample_size", type=int, default=10000, help="Sample size for Lichess dataset (default 10,000). Use -1 to evaluate all.")
    parser.add_argument("--chunk_size", type=int, default=5000, help="Chunk size for evaluation (default 5000).")
    parser.add_argument("--no_filter", action="store_true", help="Do not filter by is_puzzle or validity.")
    args = parser.parse_args()

    if not args.generated_csv and not args.lichess_csv:
        print("Error: Please provide at least one of --generated_csv or --lichess_csv.", flush=True)
        return

    # 1. Identify generated CSV files
    generated_files = []
    if args.generated_csv:
        path = Path(args.generated_csv)
        if not path.exists():
            print(f"Error: {args.generated_csv} does not exist.", flush=True)
            return        
        if path.is_dir() or not args.generated_csv.endswith(".csv"):
            if path.is_dir():
                generated_files = sorted(list(path.glob("*.csv")), key=natural_sort_key)
                if not generated_files:
                    print(f"Error: No CSV files found in directory {args.generated_csv}.", flush=True)
                    return
            else:
                print(f"Error: {args.generated_csv} is not a CSV file and is not a directory.", flush=True)
                return
        else:
            generated_files = [path]

    # 2. Load Lichess dataset
    df_lic = None
    if args.lichess_csv:
        path = Path(args.lichess_csv)
        if not path.exists():
            print(f"Error: {args.lichess_csv} does not exist.", flush=True)
            return
        print(f"Loading Lichess dataset from {path}...", flush=True)
        if args.lichess_sample_size > 0:
            nrows_to_load = args.lichess_sample_size * 2 if not args.no_filter else args.lichess_sample_size
            df_lic = pd.read_csv(path, nrows=nrows_to_load)
        else:
            df_lic = pd.read_csv(path)
        print(f"Loaded {len(df_lic)} rows from Lichess.", flush=True)

        if not args.no_filter:
            filter_col = None
            for col in ["is_puzzle_true", "is_puzzle"]:
                if col in df_lic.columns:
                    filter_col = col
                    break
            if filter_col is not None:
                df_lic = df_lic[df_lic[filter_col] == True]
                print(f"Filtered Lichess to {len(df_lic)} puzzles ({filter_col} == True).", flush=True)

    # 3. Sample and align sizes for Lichess
    lic_fens, lic_pvs = [], []
    if df_lic is not None:
        fen_col = "Puzzle_FEN" if "Puzzle_FEN" in df_lic.columns else ("FEN" if "FEN" in df_lic.columns else "fen")
        pv_col = "Moves" if "Moves" in df_lic.columns else "main_line"
        
        df_lic_valid = df_lic.dropna(subset=[fen_col, pv_col])
        n_lic = len(df_lic_valid)
        sample_n = args.lichess_sample_size if (args.lichess_sample_size > 0 and args.lichess_sample_size < n_lic) else n_lic
        df_lic_sampled = df_lic_valid.sample(sample_n) if sample_n < n_lic else df_lic_valid
        
        lic_fens = df_lic_sampled[fen_col].tolist()
        lic_pvs = [extract_pv_moves(pv, is_lichess=True) for pv in df_lic_sampled[pv_col]]
        print(f"Prepared {len(lic_fens)} Lichess samples.", flush=True)

    # 4. Iterate over generated files and compute distances
    results = []
    for gen_path in generated_files:
        print("\n" + "="*80, flush=True)
        print(f"Processing generated file: {gen_path.name}", flush=True)
        print("="*80, flush=True)

        print(f"Loading generated dataset from {gen_path}...", flush=True)
        df_gen = pd.read_csv(gen_path)
        print(f"Loaded {len(df_gen)} rows.", flush=True)

        if not args.no_filter and "is_puzzle" in df_gen.columns:
            df_gen = df_gen[df_gen["is_puzzle"] == True]
            print(f"Filtered to {len(df_gen)} puzzles (is_puzzle == True).", flush=True)

        gen_fens, gen_pvs = [], []
        if df_gen is not None:
            n_gen = len(df_gen)
            sample_n = args.self_sample_size if (args.self_sample_size > 0 and args.self_sample_size < n_gen) else n_gen
            df_gen_sampled = df_gen.sample(sample_n) if sample_n < n_gen else df_gen
            
            gen_fens = df_gen_sampled["fen"].tolist()
            gen_pvs = [extract_pv_moves(pv, is_lichess=False) for pv in df_gen_sampled["main_line"]]
            print(f"Prepared {len(gen_fens)} generated samples.", flush=True)

        print("\n" + "="*50, flush=True)
        print("STARTING DISTANCE COMPUTATIONS", flush=True)
        print("="*50, flush=True)

        match = re.search(r'\d+', gen_path.name)
        checkpoint = int(match.group()) if match else gen_path.stem

        result_dict = {
            "checkpoint": checkpoint,
            "self_distance_board": None,
            "self_distance_board_standard_error": None,
            "self_distance_pv": None,
            "self_distance_pv_standard_error": None,
            "lichess_distance_board": None,
            "lichess_distance_board_standard_error": None,
            "lichess_distance_pv": None,
            "lichess_distance_pv_standard_error": None,
        }

        # Compute Self-Distance for Generated Dataset
        if len(gen_fens) > 1:
            t0 = time.time()
            print(f"Computing generated dataset self-distances (sample size={len(gen_fens)})...", flush=True)
            gen_self_board, gen_self_pv = compute_dataset_distances(gen_fens, gen_fens, gen_pvs, gen_pvs, chunk_size=args.chunk_size, is_self=True)
            mean_self_board, se_self_board = compute_mean_and_se(gen_self_board)
            mean_self_pv, se_self_pv = compute_mean_and_se(gen_self_pv)
            print(f"Done in {time.time() - t0:.2f}s.", flush=True)
            print(f"  -> Generated Self-Distance (Board): {mean_self_board:.4f} (SE: {se_self_board:.4f})", flush=True)
            print(f"  -> Generated Self-Distance (PV):    {mean_self_pv:.4f} (SE: {se_self_pv:.4f})", flush=True)
            print("-"*50, flush=True)
            result_dict["self_distance_board"] = mean_self_board
            result_dict["self_distance_board_standard_error"] = se_self_board
            result_dict["self_distance_pv"] = mean_self_pv
            result_dict["self_distance_pv_standard_error"] = se_self_pv

        # Compute Cross-Distance (Novelty: Generated w.r.t Lichess)
        if len(gen_fens) > 0 and len(lic_fens) > 0:
            t0 = time.time()
            print(f"Computing Novelty distances (Generated w.r.t Lichess, gen_size={len(gen_fens)}, lic_size={len(lic_fens)})...", flush=True)
            novelty_board, novelty_pv = compute_dataset_distances(gen_fens, lic_fens, gen_pvs, lic_pvs, chunk_size=args.chunk_size, is_self=False)
            mean_novelty_board, se_novelty_board = compute_mean_and_se(novelty_board)
            mean_novelty_pv, se_novelty_pv = compute_mean_and_se(novelty_pv)
            print(f"Done in {time.time() - t0:.2f}s.", flush=True)
            print(f"  -> Novelty/Lichess-Distance (Board): {mean_novelty_board:.4f} (SE: {se_novelty_board:.4f})", flush=True)
            print(f"  -> Novelty/Lichess-Distance (PV):    {mean_novelty_pv:.4f} (SE: {se_novelty_pv:.4f})", flush=True)
            print("-"*50, flush=True)
            result_dict["lichess_distance_board"] = mean_novelty_board
            result_dict["lichess_distance_board_standard_error"] = se_novelty_board
            result_dict["lichess_distance_pv"] = mean_novelty_pv
            result_dict["lichess_distance_pv_standard_error"] = se_novelty_pv

        results.append(result_dict)

    # 5. Save results to CSV if we are iterating over a directory
    if args.generated_csv and Path(args.generated_csv).is_dir() and results:
        df_results = pd.DataFrame(results)
        output_csv_path = Path(args.generated_csv) / "distances.csv"
        df_results.to_csv(output_csv_path, index=False)
        print(f"\nSaved all distance results to {output_csv_path}", flush=True)

    # 6. If no generated files are provided, but Lichess is provided, compute Lichess self-distance
    if not generated_files and df_lic is not None:
        sample_n = args.self_sample_size if (args.self_sample_size > 0 and args.self_sample_size < len(lic_fens)) else len(lic_fens)
        lic_fens_sampled = lic_fens[:sample_n]
        lic_pvs_sampled = lic_pvs[:sample_n]

        if len(lic_fens_sampled) > 1:
            print("\n" + "="*80, flush=True)
            print("Computing self-distances for Lichess dataset", flush=True)
            print("="*80, flush=True)
            print("\n" + "="*50, flush=True)
            print("STARTING DISTANCE COMPUTATIONS", flush=True)
            print("="*50, flush=True)
            t0 = time.time()
            print(f"Computing Lichess dataset self-distances (sample size={len(lic_fens_sampled)})...", flush=True)
            lic_self_board, lic_self_pv = compute_dataset_distances(lic_fens_sampled, lic_fens_sampled, lic_pvs_sampled, lic_pvs_sampled, chunk_size=args.chunk_size, is_self=True)
            mean_self_board, se_self_board = compute_mean_and_se(lic_self_board)
            mean_self_pv, se_self_pv = compute_mean_and_se(lic_self_pv)
            print(f"Done in {time.time() - t0:.2f}s.", flush=True)
            print(f"  -> Lichess Self-Distance (Board): {mean_self_board:.4f} (SE: {se_self_board:.4f})", flush=True)
            print(f"  -> Lichess Self-Distance (PV):    {mean_self_pv:.4f} (SE: {se_self_pv:.4f})", flush=True)
            print("-"*50, flush=True)
        else:
            print("Lichess dataset has insufficient data for self-distance computation.", flush=True)


if __name__ == "__main__":
    main()
