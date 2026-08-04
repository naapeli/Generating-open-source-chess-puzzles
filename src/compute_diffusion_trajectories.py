import argparse
import queue
import random
import os
from pathlib import Path
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
import pandas as pd
from chess.engine import SimpleEngine

from MaskedDiffusion.model import MaskedDiffusion
from tokenization.tokenization import (
    FENTokens, board_token_2_str, enpassant_token_2_str,
    theme_preprocessor, unscale_ratings, tokens_to_fen, tokens_to_move
)
from metrics.themes import legal, get_unique_puzzle_from_fen, counter_intuitive
from metrics.cook import cook
from rl.espo import theme_reward


def tokens_to_partial_fen(tokens: torch.Tensor, mask_token: int = 72) -> str:
    """Converts tokens into a partial FEN string where unmasked symbols are shown and mask tokens are '?'."""
    tokens_list = tokens.tolist()
    
    # Board tokens (0..63)
    board_chars = []
    for t in tokens_list[:64]:
        if t == mask_token:
            board_chars.append("?")
        elif t in board_token_2_str:
            board_chars.append(board_token_2_str[t])
        else:
            raise RuntimeError(f"Unknown board token: {t}")
    
    raw_board = "".join(board_chars)
    ranks = [raw_board[i:i+8] for i in range(0, 64, 8)]
    board_str = "/".join(ranks)
    
    # Side to move (64)
    side_token = tokens_list[64]
    side = "w" if side_token == FENTokens.side_white else ("b" if side_token == FENTokens.side_black else "?")
    
    # Castling (65..68)
    castling_map = [(65, FENTokens.castle_white_king, "K"), (66, FENTokens.castle_white_queen, "Q"),
                    (67, FENTokens.castle_black_king, "k"), (68, FENTokens.castle_black_queen, "q")]
    castling_chars = []
    for idx, expected, char in castling_map:
        t = tokens_list[idx]
        if t == expected:
            castling_chars.append(char)
        elif t == mask_token:
            castling_chars.append("?")
    castling = "".join(castling_chars) if castling_chars else "-"
    
    # En passant (69..70)
    ep1, ep2 = tokens_list[69], tokens_list[70]
    if ep1 == mask_token or ep2 == mask_token:
        ep = "??"
    elif ep1 in enpassant_token_2_str and ep2 in enpassant_token_2_str and ep1 != FENTokens.none and ep2 != FENTokens.none:
        ep = enpassant_token_2_str[ep1] + enpassant_token_2_str[ep2]
    else:
        ep = "-"
        
    return f"{board_str} {side} {castling} {ep}"


@torch.no_grad()
def sample_partial(
    model,
    tokens: torch.Tensor,
    start_step_idx: int,
    end_step_idx: int,
    steps: int = 256,
    temperature: float = 1.0,
    theme_tokens: torch.Tensor = None,
    ratings: torch.Tensor = None,
    generate_move_last: bool = True
) -> torch.Tensor:
    """Runs discrete masked diffusion transition steps from start_step_idx down to end_step_idx."""
    device = tokens.device
    mask_token = model.config.mask_token
    batch_size, seq_length = tokens.shape
    
    T_grid = torch.linspace(0, 1, steps + 1, device=device)
    
    if not model.config.predict_moves:
        generate_move_last = False
        
    if generate_move_last:
        phases = [(0, model.config.fen_length, steps), (model.config.fen_length, seq_length, steps // 4)]
    else:
        phases = [(0, seq_length, steps)]
        
    for start_idx, end_idx, step_count in phases:
        effective_start = min(start_step_idx, step_count)
        effective_end = max(end_step_idx, 0)
        
        if effective_start <= effective_end:
            continue
            
        for i in range(effective_start, effective_end, -1):
            t = T_grid[i]
            s = T_grid[i - 1]
            
            alpha_t = model.config.masking_schedule(t)
            alpha_s = model.config.masking_schedule(s)
            if s == 0.0:
                alpha_s = torch.ones_like(alpha_s)
                
            logits = model(tokens, theme_tokens, ratings)
            probs = F.softmax(logits / temperature, dim=2)
            
            p_unmask = (alpha_s - alpha_t) / (1.0 - alpha_t + 1e-13)
            p_mask = (1.0 - alpha_s) / (1.0 - alpha_t + 1e-13)
            
            probs = torch.cat([probs * p_unmask, torch.full((batch_size, seq_length, 1), p_mask, device=device, dtype=probs.dtype)], dim=2)
            
            log_probs = torch.log(probs + 1e-13)
            u = torch.rand_like(log_probs)
            gumbel_noise = -torch.log(-torch.log(u + 1e-13) + 1e-13)
            new_samples = torch.argmax(log_probs + gumbel_noise, dim=-1)
            
            is_masked = (tokens == mask_token)
            in_window = torch.zeros_like(is_masked, dtype=torch.bool)
            in_window[:, start_idx:end_idx] = True
            is_updatable = is_masked & in_window
            
            tokens = torch.where(is_updatable, new_samples, tokens)
            
    return tokens


def evaluate_leaf_puzzle(leaf_info: dict, base_theme: list, engine_pool: queue.Queue, config) -> dict:
    """Evaluates puzzle criteria using Stockfish ONLY for final leaf positions at step 0."""
    tokens = leaf_info["tokens"]
    fen_tokens = tokens[:config.fen_length]
    
    result = {
        "leaf_node_id": leaf_info["node_id"],
        "fen": None,
        "is_legal": False,
        "is_puzzle": False,
        "best_move": None,
        "main_line": None,
        "actual_themes": None,
        "themes_match": None,
        "counter_intuitive_value": None,
    }
    
    try:
        fen = tokens_to_fen(fen_tokens)
        result["fen"] = fen
        if config.predict_moves and len(tokens) > config.fen_length:
            move_tokens = tokens[config.fen_length:]
            result["best_move"] = tokens_to_move(move_tokens)
    except Exception:
        return result
        
    if not legal(fen):
        return result
        
    result["is_legal"] = True
    
    engine = engine_pool.get()
    try:
        engine.configure({"Clear Hash": None})
        _, ci_val = counter_intuitive(fen, engine, return_value=True)
        result["counter_intuitive_value"] = float(ci_val)
        
        puzzle = get_unique_puzzle_from_fen(fen, engine)
        if puzzle is not None:
            result["is_puzzle"] = True
            result["main_line"] = " ".join([move.uci() for move in puzzle.mainline])
            if len(puzzle.mainline) > 0 and result["best_move"] is None:
                result["best_move"] = puzzle.mainline[0].uci()
            existing_themes = cook(puzzle, engine)
            result["actual_themes"] = existing_themes
            
            if config.use_context and base_theme is not None:
                result["themes_match"] = bool(theme_reward(base_theme, existing_themes))
    except Exception:
        pass
    finally:
        engine_pool.put(engine)
        
    return result


def flush_rows_to_csv(rows: list, output_path: Path):
    """Appends accumulated trajectory rows to the target CSV file."""
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    file_exists = output_path.exists()
    df.to_csv(output_path, mode="a" if file_exists else "w", header=not file_exists, index=False)


def main():
    parser = argparse.ArgumentParser(description="Generate trajectory trees with multi-tree GPU batching and export results to CSV.")
    parser.add_argument("--run_type", choices=["supervised", "rl"], required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--checkpoint_name", type=str, required=True)
    parser.add_argument("--n_trees", type=int, default=150, help="Total number of trajectory trees to generate.")
    parser.add_argument("--tree_batch_size", type=int, default=128, help="Number of trees to process in parallel on GPU.")
    parser.add_argument("--steps", type=int, default=256, help="Total diffusion steps.")
    parser.add_argument("--n_milestones", type=int, default=4, help="Number of intermediate milestone intervals.")
    parser.add_argument("--branching_factor", type=int, default=3, help="Number of child branches per node at each milestone.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output_file", type=str, default="diffusion_trajectories.csv", help="Output CSV filename.")
    parser.add_argument("--target_themes", nargs="+", default=None, help="Specific theme(s) to require in target conditioning (e.g. --target_themes fork pin).")
    parser.add_argument("--exclude_themes", nargs="+", default=None, help="Theme(s) to exclude from target conditioning (e.g. --exclude_themes quietMove master).")
    parser.add_argument("--context_dataset", choices=["train", "test"], default="test")
    args = parser.parse_args()

    base_path = Path("./src")
    checkpoint_path = base_path / "runs" / args.run_type / args.run_name / args.checkpoint_name
    print(f"Loading model checkpoint from {checkpoint_path}...", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    config = checkpoint["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MaskedDiffusion(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device=device)
    model.eval()

    milestone_steps = [int(round(x)) for x in torch.linspace(args.steps, 0, args.n_milestones + 1).tolist()]
    print(f"Milestone step schedule: {milestone_steps}", flush=True)

    slurm_cpus = os.getenv("SLURM_CPUS_PER_GPU")
    n_jobs = int(slurm_cpus) - 2 if slurm_cpus is not None else max(1, (os.cpu_count() or 4) - 2)
    stockfish_path = base_path / ".." / "Stockfish" / "src" / "stockfish"
    
    engine_pool = queue.Queue()
    print(f"Initializing {n_jobs} Stockfish engine workers...", flush=True)
    for _ in range(n_jobs):
        engine = SimpleEngine.popen_uci(stockfish_path)
        engine.configure({"Threads": 1, "Hash": 32})
        engine_pool.put(engine)

    if config.use_context:
        dataset_name = "trainset.pt" if args.context_dataset == "train" else "testset.pt"
        dataset_path = base_path / "dataset" / "with_best_move" / dataset_name
        raw_dataset = torch.load(dataset_path, weights_only=False, map_location="cpu")
        
        target_set = set(args.target_themes) if args.target_themes else None
        exclude_set = set(args.exclude_themes) if args.exclude_themes else None

        if target_set or exclude_set:
            filtered_dataset = []
            for item in raw_dataset:
                sampled_theme = torch.as_tensor(item[2])
                item_themes = set(theme_preprocessor.inverse_transform(sampled_theme.reshape(1, -1).numpy())[0])
                
                if target_set and not target_set.issubset(item_themes):
                    continue
                if exclude_set and item_themes.intersection(exclude_set):
                    continue
                    
                filtered_dataset.append(item)
                
            if not filtered_dataset:
                raise ValueError(f"No items in dataset match target_themes={args.target_themes} and exclude_themes={args.exclude_themes}")
                
            print(f"Filtered context dataset from {len(raw_dataset)} to {len(filtered_dataset)} items matching theme criteria.", flush=True)
            dataset = filtered_dataset
        else:
            dataset = raw_dataset
    else:
        dataset = None

    seq_length = model.seq_length
    mask_token = config.mask_token
    output_path = base_path / "Generate_positions" / args.output_file

    buffer_rows = []
    total_saved_trajectories = 0
    executor = ThreadPoolExecutor(max_workers=n_jobs)

    try:
        for chunk_start in range(0, args.n_trees, args.tree_batch_size):
            chunk_end = min(chunk_start + args.tree_batch_size, args.n_trees)
            current_tree_batch_size = chunk_end - chunk_start
            print(f"\n=== Processing Tree Batch: Trees {chunk_start + 1} to {chunk_end} (Parallel Trees: {current_tree_batch_size}) ===", flush=True)

            tree_contexts = []
            for b_idx in range(current_tree_batch_size):
                tree_idx = chunk_start + b_idx
                if config.use_context:
                    rand_idx = random.randint(0, len(dataset) - 1)
                    sampled_item = dataset[rand_idx]
                    sampled_theme = torch.as_tensor(sampled_item[2])
                    sampled_rating = torch.as_tensor(sampled_item[3])
                    
                    base_theme = theme_preprocessor.inverse_transform(sampled_theme.reshape(1, -1).numpy())[0]
                    base_rating = unscale_ratings(sampled_rating).item()
                    
                    theme_tensor = sampled_theme.reshape(1, -1).to(device=device, dtype=torch.float32)
                    rating_tensor = sampled_rating.reshape(1).to(device=device, dtype=torch.float32)
                else:
                    base_theme = None
                    base_rating = None
                    theme_tensor = None
                    rating_tensor = None

                tree_contexts.append({
                    "tree_idx": tree_idx,
                    "base_theme": base_theme,
                    "base_rating": base_rating,
                    "theme_tensor": theme_tensor,
                    "rating_tensor": rating_tensor,
                    "tree_nodes": {},
                    "frontier": ["0"],
                })

            root_tokens = torch.full((1, seq_length), mask_token, device=device, dtype=torch.long)
            for ctx in tree_contexts:
                ctx["tree_nodes"]["0"] = {
                    "node_id": "0",
                    "parent_id": None,
                    "step": milestone_steps[0],
                    "tokens": root_tokens[0].cpu(),
                }

            # Step through milestones for ALL trees in GPU batch simultaneously
            for m in range(len(milestone_steps) - 1):
                start_step = milestone_steps[m]
                end_step = milestone_steps[m + 1]
                
                batch_tokens_list = []
                batch_themes_list = []
                batch_ratings_list = []
                tree_slice_info = []

                for ctx_idx, ctx in enumerate(tree_contexts):
                    frontier = ctx["frontier"]
                    num_parents = len(frontier)
                    num_children = num_parents * args.branching_factor
                    
                    parent_tokens = torch.stack([ctx["tree_nodes"][pid]["tokens"] for pid in frontier]).to(device=device)
                    child_tokens = parent_tokens.repeat_interleave(args.branching_factor, dim=0)
                    batch_tokens_list.append(child_tokens)

                    if ctx["theme_tensor"] is not None:
                        batch_themes_list.append(ctx["theme_tensor"].expand(num_children, -1))
                        batch_ratings_list.append(ctx["rating_tensor"].expand(num_children))

                    child_ids = []
                    for idx, pid in enumerate(frontier):
                        for b in range(args.branching_factor):
                            child_ids.append((pid, f"{pid}_{b}"))
                    tree_slice_info.append((ctx_idx, num_children, child_ids))

                combined_tokens = torch.cat(batch_tokens_list, dim=0)
                combined_themes = torch.cat(batch_themes_list, dim=0) if batch_themes_list else None
                combined_ratings = torch.cat(batch_ratings_list, dim=0) if batch_ratings_list else None

                print(f"  Milestone {start_step} -> {end_step}: Running GPU sampling for batch of {combined_tokens.shape[0]} sequences...", flush=True)
                
                start_time = perf_counter()
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    updated_combined_tokens = sample_partial(
                        model,
                        combined_tokens,
                        start_step_idx=start_step,
                        end_step_idx=end_step,
                        steps=args.steps,
                        temperature=args.temperature,
                        theme_tokens=combined_themes,
                        ratings=combined_ratings
                    )
                sampling_time = perf_counter() - start_time
                print(f"    GPU Sampling completed in {sampling_time:.3f}s", flush=True)

                updated_combined_cpu = updated_combined_tokens.cpu()

                # Split updated tokens back to respective trees
                offset = 0
                for ctx_idx, num_children, child_ids in tree_slice_info:
                    tree_updated = updated_combined_cpu[offset : offset + num_children]
                    offset += num_children
                    
                    ctx = tree_contexts[ctx_idx]
                    next_frontier = []
                    for child_idx, (pid, child_node_id) in enumerate(child_ids):
                        ctx["tree_nodes"][child_node_id] = {
                            "node_id": child_node_id,
                            "parent_id": pid,
                            "step": end_step,
                            "tokens": tree_updated[child_idx],
                        }
                        next_frontier.append(child_node_id)
                    ctx["frontier"] = next_frontier

            # Evaluate Stockfish metrics for ALL final leaf positions in batch
            all_leaf_tasks = []
            for ctx in tree_contexts:
                leaf_nodes = [ctx["tree_nodes"][nid] for nid in ctx["frontier"]]
                for leaf_info in leaf_nodes:
                    all_leaf_tasks.append((ctx, leaf_info))

            print(f"  Evaluating {len(all_leaf_tasks)} final leaf positions with Stockfish engine pool...", flush=True)
            eval_futures = [
                (ctx, executor.submit(evaluate_leaf_puzzle, leaf_info, ctx["base_theme"], engine_pool, config))
                for ctx, leaf_info in all_leaf_tasks
            ]

            # Reconstruct trajectories and build CSV rows for this batch
            for ctx, fut in eval_futures:
                leaf_eval = fut.result()
                trajectory_steps = []
                curr_id = leaf_eval["leaf_node_id"]
                while curr_id is not None:
                    n_info = ctx["tree_nodes"][curr_id]
                    unmasked_c = (n_info["tokens"] != mask_token).sum().item()
                    p_fen = tokens_to_partial_fen(n_info["tokens"], mask_token)
                    trajectory_steps.append({
                        "step": n_info["step"],
                        "node_id": n_info["node_id"],
                        "unmasked_count": unmasked_c,
                        "partial_fen": p_fen
                    })
                    curr_id = n_info["parent_id"]
                    
                trajectory_steps.reverse()

                row = {
                    "tree_id": ctx["tree_idx"],
                    "trajectory_id": f"tree_{ctx['tree_idx']}_{leaf_eval['leaf_node_id']}",
                    "target_themes": str(ctx["base_theme"]) if ctx["base_theme"] is not None else None,
                    "target_rating": ctx["base_rating"],
                    "fen": leaf_eval["fen"],
                    "is_legal": leaf_eval["is_legal"],
                    "is_puzzle": leaf_eval["is_puzzle"],
                    "best_move": leaf_eval["best_move"],
                    "main_line": leaf_eval["main_line"],
                    "actual_themes": str(leaf_eval["actual_themes"]),
                    "themes_match": leaf_eval["themes_match"],
                    "counter_intuitive_value": leaf_eval["counter_intuitive_value"],
                }
                for step_info in trajectory_steps:
                    step_num = step_info["step"]
                    row[f"node_id_step_{step_num}"] = step_info["node_id"]
                    row[f"partial_fen_step_{step_num}"] = step_info["partial_fen"]
                    row[f"unmasked_count_step_{step_num}"] = step_info["unmasked_count"]
                
                buffer_rows.append(row)

            # Flush completed batch to CSV
            flush_rows_to_csv(buffer_rows, output_path)
            total_saved_trajectories += len(buffer_rows)
            print(f"  --> Appended {len(buffer_rows)} trajectories from batch to CSV (Total saved: {total_saved_trajectories})", flush=True)
            buffer_rows.clear()

    finally:
        while not engine_pool.empty():
            eng = engine_pool.get()
            eng.quit()
        executor.shutdown()

    if buffer_rows:
        flush_rows_to_csv(buffer_rows, output_path)
        total_saved_trajectories += len(buffer_rows)
        buffer_rows.clear()

    print(f"\nCompleted run! Total {total_saved_trajectories} trajectory rows written to: {output_path}")


if __name__ == "__main__":
    main()
