import argparse
import queue
import os
from pathlib import Path
import pandas as pd
from chess.engine import SimpleEngine
from joblib import Parallel, delayed
from time import perf_counter

from metrics.themes import legal, get_unique_puzzle_from_fen, counter_intuitive


def process_row(puzzle_fen, rating, themes, engine_pool):
    entry = {
        "fen": puzzle_fen,
        "target_rating": rating,
        "target_themes": themes,
        "is_legal": False,
        "is_puzzle": False,
        "counter_intuitive": None,
    }

    engine = engine_pool.get()

    try:
        if not legal(puzzle_fen):
            return entry

        entry["is_legal"] = True

        engine.configure({"Clear Hash": None})
        entry["counter_intuitive"] = counter_intuitive(puzzle_fen, engine)
        puzzle = get_unique_puzzle_from_fen(puzzle_fen, engine)
        if puzzle is not None:
            entry["is_puzzle"] = True

        return entry

    except Exception as e:
        print(f"Error processing FEN {puzzle_fen}: {e}", flush=True)
        return entry
    finally:
        engine_pool.put(engine)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file",
        type=str,
        default="lichess_evaluated.csv",
        help="Output filename inside src/Generate_positions/",
    )
    parser.add_argument(
        "--n_puzzles",
        type=int,
        default=10000,
        help="Number of puzzles to evaluate from dataset",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=None,
        help="Number of parallel jobs (default: CPU count - 2 or SLURM cpus)",
    )
    args = parser.parse_args()

    base_path = Path("./src")
    input_path = base_path / "dataset" / "dataset.csv"
    stockfish_path = base_path / ".." / "Stockfish" / "src" / "stockfish"

    if args.n_jobs is None:
        slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK") or os.getenv(
            "SLURM_CPUS_PER_GPU"
        )
        if slurm_cpus:
            args.n_jobs = max(1, int(slurm_cpus) - 2)
        else:
            args.n_jobs = max(1, os.cpu_count() - 2)

    print(
        f"Loading first {args.n_puzzles} puzzles from {input_path}...",
        flush=True,
    )
    start_load = perf_counter()
    df = pd.read_csv(input_path, nrows=args.n_puzzles)
    print(f"Loaded in {perf_counter() - start_load:.2f} seconds.", flush=True)

    print(
        f"Initializing engine pool with {args.n_jobs} jobs...", flush=True
    )
    engine_pool = queue.Queue()
    for _ in range(args.n_jobs):
        engine = SimpleEngine.popen_uci(stockfish_path)
        engine.configure({"Threads": 1, "Hash": 32})
        engine_pool.put(engine)

    print("Evaluating puzzles...", flush=True)
    start_eval = perf_counter()

    rows = df.to_dict(orient="records")

    results = Parallel(n_jobs=args.n_jobs, backend="threading")(
        delayed(process_row)(
            row["Puzzle_FEN"],
            int(row["Rating"]) if not pd.isna(row["Rating"]) else None,
            (
                row["Themes"].split(" ")
                if isinstance(row["Themes"], str)
                else []
            ),
            engine_pool,
        )
        for row in rows
    )

    print(
        f"Evaluated all puzzles in {perf_counter() - start_eval:.2f} seconds.",
        flush=True,
    )

    # Clean up engine pool
    while not engine_pool.empty():
        engine = engine_pool.get()
        engine.quit()

    # Save to dataframe
    out_df = pd.DataFrame(results)

    output_dir = base_path / "Generate_positions"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_file

    out_df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}", flush=True)


if __name__ == "__main__":
    main()
