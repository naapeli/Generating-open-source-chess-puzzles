import argparse
import os
import queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
import pandas as pd
from chess.engine import SimpleEngine

from metrics.themes import legal, uniqueness, counter_intuitive

engine_pool = queue.Queue()


def process_row(row_dict, fen_col):
    entry = dict(row_dict)
    puzzle_fen = (
        str(entry[fen_col])
        if fen_col in entry and not pd.isna(entry[fen_col])
        else ""
    )

    entry["is_legal"] = False
    entry["is_puzzle_true"] = False
    entry["counter_intuitive_true"] = False

    engine = engine_pool.get()

    try:
        if not legal(puzzle_fen):
            return entry

        entry["is_legal"] = True

        engine.configure({"Clear Hash": None})
        is_ci = counter_intuitive(puzzle_fen, engine)
        entry["counter_intuitive_true"] = is_ci

        is_uniq = uniqueness(puzzle_fen, engine)
        entry["is_puzzle_true"] = is_uniq

        return entry

    except Exception as e:
        print(f"Error processing FEN {puzzle_fen}: {e}", flush=True)
        return entry
    finally:
        engine_pool.put(engine)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to input CSV dataset file (defaults to src/dataset/dataset.csv)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="lichess_evaluated.csv",
        help="Output filename inside src/Generate_positions/ (or explicit path)",
    )
    parser.add_argument(
        "--n_puzzles",
        type=int,
        default=10000,
        help="Number of puzzles to evaluate from dataset",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Batch size for incremental evaluation and checkpoint saving",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=None,
        help="Number of parallel jobs (default: CPU count - 2 or SLURM cpus)",
    )
    args = parser.parse_args()

    base_path = Path("./src")

    if args.input_file:
        p = Path(args.input_file)
        if p.exists():
            input_path = p
        elif (base_path / args.input_file).exists():
            input_path = base_path / args.input_file
        elif (base_path / "dataset" / args.input_file).exists():
            input_path = base_path / "dataset" / args.input_file
        else:
            input_path = p
    else:
        input_path = base_path / "dataset" / "dataset.csv"

    # Resolve output path
    out_file_path = Path(args.output_file)
    if out_file_path.is_absolute():
        out_base = out_file_path
    else:
        output_dir = base_path / "Generate_positions"
        out_base = output_dir / args.output_file

    if input_path.is_dir():
        input_files = sorted([f for f in input_path.glob("*.csv") if f.name != "distances.csv"])
        if not input_files:
            raise FileNotFoundError(f"No .csv files found in directory {input_path}")
        out_dir = out_base.parent if out_base.suffix == '.csv' else out_base
        files_to_process = [(f_in, out_dir / f_in.name) for f_in in input_files]
    else:
        files_to_process = [(input_path, out_base)]

    stockfish_path = base_path / ".." / "Stockfish" / "src" / "stockfish"

    if args.n_jobs is None:
        slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK") or os.getenv(
            "SLURM_CPUS_PER_GPU"
        )
        if slurm_cpus:
            args.n_jobs = max(1, int(slurm_cpus) - 2)
        else:
            args.n_jobs = max(1, os.cpu_count() - 2)

    # Initialize engine pool once
    print(f"Initializing engine pool with {args.n_jobs} jobs...", flush=True)
    for _ in range(args.n_jobs):
        engine = SimpleEngine.popen_uci(stockfish_path)
        engine.configure({"Threads": 1, "Hash": 32})
        engine_pool.put(engine)

    try:
        for f_in, f_out in files_to_process:
            print(f"\nProcessing {f_in} -> {f_out}...", flush=True)
            f_out.parent.mkdir(parents=True, exist_ok=True)

            try:
                df = pd.read_csv(f_in)
            except Exception as e:
                print(f"Error reading CSV file {f_in}: {e}. Skipping.", flush=True)
                continue

            if args.n_puzzles is not None and args.n_puzzles > 0 and len(df) > args.n_puzzles:
                df = df.head(args.n_puzzles)

            fen_col = None
            for col in ["fen", "Puzzle_FEN", "FEN", "puzzle_fen"]:
                if col in df.columns:
                    fen_col = col
                    break

            if fen_col is None:
                fen_cols = [c for c in df.columns if "fen" in c.lower()]
                if fen_cols:
                    fen_col = fen_cols[0]
                else:
                    print(
                        f"Warning: Could not find a 'fen' or 'Puzzle_FEN' column in {f_in}. Available columns: {list(df.columns)}. Skipping.",
                        flush=True,
                    )
                    continue

            # Check for existing checkpoint to resume from
            start_idx = 0
            if f_out.exists() and f_out.stat().st_size > 0:
                try:
                    existing_df = pd.read_csv(f_out)
                    start_idx = len(existing_df)
                    print(
                        f"Found existing output file {f_out} with {start_idx} evaluated rows. Resuming...",
                        flush=True,
                    )
                except Exception as e:
                    print(
                        f"Warning: Could not read existing file {f_out}: {e}. Starting from scratch.",
                        flush=True,
                    )
                    start_idx = 0

            if start_idx >= len(df):
                print(
                    f"All {len(df)} puzzles have already been evaluated and saved in {f_out}.",
                    flush=True,
                )
                continue

            remaining_rows = df.iloc[start_idx:].to_dict(orient="records")
            total_remaining = len(remaining_rows)

            batch_size = args.batch_size
            num_batches = (total_remaining + batch_size - 1) // batch_size
            print(
                f"Evaluating remaining {total_remaining} puzzles in {num_batches} batches of up to {batch_size}...",
                flush=True,
            )
            start_eval = perf_counter()

            executor = ThreadPoolExecutor(max_workers=args.n_jobs)
            processed_count = start_idx

            try:
                for b in range(num_batches):
                    batch_rows = remaining_rows[b * batch_size : (b + 1) * batch_size]
                    batch_args = [(row, fen_col) for row in batch_rows]

                    b_start = perf_counter()
                    batch_results = list(executor.map(lambda p: process_row(*p), batch_args))
                    b_time = perf_counter() - b_start

                    df_batch = pd.DataFrame(batch_results)

                    write_header = (
                        not f_out.exists() or f_out.stat().st_size == 0
                    )
                    df_batch.to_csv(f_out, mode="a", index=False, header=write_header)

                    processed_count += len(batch_results)
                    print(
                        f"Batch {b + 1}/{num_batches} finished in {b_time:.2f}s. Total saved: {processed_count}/{len(df)} -> {f_out}",
                        flush=True,
                    )
            finally:
                executor.shutdown(wait=True)

            print(
                f"Finished evaluating {f_in} in {perf_counter() - start_eval:.2f} seconds.",
                flush=True,
            )
    finally:
        # Clean up engine pool
        print("Cleaning up engine pool...", flush=True)
        while not engine_pool.empty():
            engine = engine_pool.get()
            engine.quit()


if __name__ == "__main__":
    main()


