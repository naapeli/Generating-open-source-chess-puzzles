import argparse
import queue
import re
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
    FENTokens, board_token_2_str, board_str_2_token, enpassant_token_2_str, enpassant_str_2_token,
    promote_str_2_token, counter_str_2_token, theme_preprocessor, scale_ratings, unscale_ratings,
    tokens_to_fen, tokens_to_move
)
from metrics.themes import legal, get_unique_puzzle_from_fen, counter_intuitive
from metrics.cook import cook
from rl.espo import theme_reward


def partial_fen_to_tokens(partial_fen: str, config, mask_token: int = 72, best_move: str = None) -> torch.Tensor:
    """
    Parses a partial or full FEN string into a seq_length token tensor.
    Unmasked pieces/symbols map to FENTokens, '?' maps to mask_token (72).
    Supports '?' for board, side to move, castling, en passant, halfmove, fullmove, and move tokens.
    """
    parts = partial_fen.strip().split(" ")
    board_part = parts[0]
    side_part = parts[1] if len(parts) > 1 else "w"
    castling_part = parts[2] if len(parts) > 2 else "KQkq"
    ep_part = parts[3] if len(parts) > 3 else "-"
    halfmove_part = parts[4] if len(parts) > 4 else "0"
    fullmove_part = parts[5] if len(parts) > 5 else "1"

    # 1. Expand digits to dots
    expanded_board = re.sub(r"\d", lambda digit: "." * int(digit.group(0)), board_part)
    expanded_board = expanded_board.replace("/", "")

    if len(expanded_board) != 64:
        raise ValueError(f"Invalid partial board length after expansion: {len(expanded_board)} (expected 64)")

    tokens_list = []
    for char in expanded_board:
        if char == "?":
            tokens_list.append(mask_token)
        elif char == ".":
            tokens_list.append(FENTokens.no_piece)
        elif char in board_str_2_token:
            tokens_list.append(board_str_2_token[char])
        else:
            raise ValueError(f"Unknown board character: '{char}'")

    # 2. Side to move (64)
    if side_part == "w":
        tokens_list.append(FENTokens.side_white)
    elif side_part == "b":
        tokens_list.append(FENTokens.side_black)
    elif "?" in side_part:
        tokens_list.append(mask_token)
    else:
        tokens_list.append(FENTokens.side_white)

    # 3. Castling (65..68)
    castling_chars = "KQkq"
    castling_tokens_map = [
        FENTokens.castle_white_king, FENTokens.castle_white_queen,
        FENTokens.castle_black_king, FENTokens.castle_black_queen
    ]
    if castling_part == "-":
        tokens_list.extend([FENTokens.no_castle] * 4)
    elif castling_part == "?":
        tokens_list.extend([mask_token] * 4)
    elif castling_part == "?-":
        tokens_list.extend([mask_token, mask_token, FENTokens.no_castle, FENTokens.no_castle])
    elif castling_part == "-?":
        tokens_list.extend([FENTokens.no_castle, FENTokens.no_castle, mask_token, mask_token])
    elif len(castling_part) == 4:
        for idx, (c, tok) in enumerate(zip(castling_chars, castling_tokens_map)):
            char = castling_part[idx]
            if char == c:
                tokens_list.append(tok)
            elif char == "?":
                tokens_list.append(mask_token)
            else:
                tokens_list.append(FENTokens.no_castle)
    else:
        has_q_mark = "?" in castling_part
        has_w_char = any(c in castling_part for c in "KQ")
        has_b_char = any(c in castling_part for c in "kq")
        
        # White Kingside (K)
        if "K" in castling_part:
            tokens_list.append(FENTokens.castle_white_king)
        elif has_q_mark and not has_w_char:
            tokens_list.append(mask_token)
        else:
            tokens_list.append(FENTokens.no_castle)
            
        # White Queenside (Q)
        if "Q" in castling_part:
            tokens_list.append(FENTokens.castle_white_queen)
        elif has_q_mark and not has_w_char:
            tokens_list.append(mask_token)
        else:
            tokens_list.append(FENTokens.no_castle)

        # Black Kingside (k)
        if "k" in castling_part:
            tokens_list.append(FENTokens.castle_black_king)
        elif has_q_mark and not has_b_char:
            tokens_list.append(mask_token)
        else:
            tokens_list.append(FENTokens.no_castle)

        # Black Queenside (q)
        if "q" in castling_part:
            tokens_list.append(FENTokens.castle_black_queen)
        elif has_q_mark and not has_b_char:
            tokens_list.append(mask_token)
        else:
            tokens_list.append(FENTokens.no_castle)

    # 4. En passant (69..70)
    if "?" in ep_part:
        if len(ep_part) == 1:
            tokens_list.extend([mask_token, mask_token])
        else:
            f_tok = mask_token if ep_part[0] == "?" else enpassant_str_2_token.get(ep_part[0], FENTokens.none)
            r_tok = mask_token if ep_part[1] == "?" else enpassant_str_2_token.get(ep_part[1], FENTokens.none)
            tokens_list.extend([f_tok, r_tok])
    elif ep_part != "-" and len(ep_part) == 2:
        file_tok = enpassant_str_2_token.get(ep_part[0], FENTokens.none)
        rank_tok = enpassant_str_2_token.get(ep_part[1], FENTokens.none)
        tokens_list.extend([file_tok, rank_tok])
    else:
        tokens_list.extend([FENTokens.none, FENTokens.none])

    # 5. Halfmove counter (71..72)
    if "?" in halfmove_part:
        tokens_list.extend([mask_token, mask_token])
    else:
        hm_str = "." + halfmove_part if len(halfmove_part) == 1 else halfmove_part
        hm_toks = [counter_str_2_token.get(c, FENTokens.pad_counter) for c in hm_str[:2]]
        tokens_list.extend(hm_toks)

    # 6. Fullmove counter (73..75)
    if "?" in fullmove_part:
        tokens_list.extend([mask_token, mask_token, mask_token])
    else:
        fm_str = ".." + fullmove_part if len(fullmove_part) == 1 else ("." + fullmove_part if len(fullmove_part) == 2 else fullmove_part)
        fm_toks = [counter_str_2_token.get(c, FENTokens.pad_counter) for c in fm_str[:3]]
        tokens_list.extend(fm_toks)

    # 7. Optional move tokens if predict_moves
    if config.predict_moves:
        if best_move is not None:
            move_toks = []
            for idx, c in enumerate(best_move):
                if c == "?":
                    move_toks.append(mask_token)
                elif idx < 4:
                    move_toks.append(enpassant_str_2_token.get(c, mask_token))
                else:
                    move_toks.append(promote_str_2_token.get(c, FENTokens.none))
            while len(move_toks) < 5:
                move_toks.append(FENTokens.none)
            tokens_list.extend(move_toks[:5])
        else:
            tokens_list.extend([mask_token] * 5)

    return torch.tensor(tokens_list, dtype=torch.long)


def tokens_to_partial_fen(tokens: torch.Tensor, mask_token: int = 72) -> str:
    """Converts tokens into a partial FEN string where unmasked symbols are shown and mask tokens are '?'."""
    tokens_list = tokens.tolist()
    
    board_chars = []
    for t in tokens_list[:64]:
        if t == mask_token:
            board_chars.append("?")
        elif t in board_token_2_str:
            board_chars.append(board_token_2_str[t])
        else:
            board_chars.append("?")
    
    raw_board = "".join(board_chars)
    ranks = [raw_board[i:i+8] for i in range(0, 64, 8)]
    board_str = "/".join(ranks)
    
    side_token = tokens_list[64]
    side = "w" if side_token == FENTokens.side_white else ("b" if side_token == FENTokens.side_black else "?")
    
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
    parser = argparse.ArgumentParser(description="Sample positions starting from a custom partial board position.")
    parser.add_argument("--run_type", choices=["supervised", "rl"], required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--checkpoint_name", type=str, required=True)
    parser.add_argument("--initial_fen", type=str, required=True, help="Partial or full FEN string with '?' for masked squares.")
    parser.add_argument("--start_step", type=int, default=None, help="Starting step count (default: auto-calculated from ? mask count or config steps).")
    parser.add_argument("--n_positions", type=int, default=500, help="Total number of positions to generate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Number of positions to sample in parallel on GPU.")
    parser.add_argument("--steps", type=int, default=256, help="Total diffusion steps.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--themes", nargs="+", default=None, help="Custom theme(s) (e.g. --themes fork pin) or omit/set 'random' to sample.")
    parser.add_argument("--target_rating", type=float, default=1500.0, help="Target puzzle rating when custom themes are specified.")
    parser.add_argument("--best_move", type=str, default=None, help="Optional target best move to condition on (e.g. 'e2e4' or 'e7e8q').")
    parser.add_argument("--exclude_themes", nargs="+", default=None, help="Theme(s) to exclude when sampling random themes.")
    parser.add_argument("--output_file", type=str, default="partial_board_positions.csv", help="Output CSV filename.")
    parser.add_argument("--save_every", type=int, default=10, help="Save interval for logging progress.")
    parser.add_argument("--context_dataset", choices=["train", "test"], default="test")
    args = parser.parse_args()

    base_path = Path("./src")
    checkpoint_path = base_path / "runs" / args.run_type / args.run_name / args.checkpoint_name
    print(f"Loading model checkpoint from {checkpoint_path}...", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    config = checkpoint["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.best_move is not None and not config.predict_moves:
        raise ValueError("Cannot condition on --best_move: the loaded model checkpoint was not trained with move prediction (config.predict_moves is False).")

    model = MaskedDiffusion(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device=device)
    model.eval()

    mask_token = config.mask_token

    # Parse initial partial FEN string & optional best_move
    initial_tokens = partial_fen_to_tokens(args.initial_fen, config, mask_token, best_move=args.best_move)
    num_masked_board = (initial_tokens[:64] == mask_token).sum().item()
    
    if args.start_step is not None:
        start_step = args.start_step
    else:
        start_step = args.steps
        
    print(f"Initial partial position parsed: '{args.initial_fen}'")
    print(f"Board masked squares: {num_masked_board}/64.")
    print(f"Starting diffusion from step {start_step} down to 0 (Total steps: {args.steps}).", flush=True)

    slurm_cpus = os.getenv("SLURM_CPUS_PER_GPU")
    n_jobs = int(slurm_cpus) - 2 if slurm_cpus is not None else max(1, (os.cpu_count() or 4) - 2)
    stockfish_path = base_path / ".." / "Stockfish" / "src" / "stockfish"
    
    engine_pool = queue.Queue()
    print(f"Initializing {n_jobs} Stockfish engine workers...", flush=True)
    for _ in range(n_jobs):
        engine = SimpleEngine.popen_uci(stockfish_path)
        engine.configure({"Threads": 1, "Hash": 32})
        engine_pool.put(engine)

    # Context dataset for random theme sampling if custom themes not provided
    if config.use_context:
        dataset_name = "trainset.pt" if args.context_dataset == "train" else "testset.pt"
        dataset_path = base_path / "dataset" / "with_best_move" / dataset_name
        raw_dataset = torch.load(dataset_path, weights_only=False, map_location="cpu")
        
        exclude_set = set(args.exclude_themes) if args.exclude_themes else None
        if exclude_set:
            filtered_dataset = []
            for item in raw_dataset:
                sampled_theme = torch.as_tensor(item[2])
                item_themes = set(theme_preprocessor.inverse_transform(sampled_theme.reshape(1, -1).numpy())[0])
                if not item_themes.intersection(exclude_set):
                    filtered_dataset.append(item)
            dataset = filtered_dataset
        else:
            dataset = raw_dataset
    else:
        dataset = None

    output_path = base_path / "Generate_positions" / args.output_file

    use_custom_themes = (args.themes is not None and args.themes != ["random"])
    if use_custom_themes and config.use_context:
        custom_theme_one_hot = theme_preprocessor.transform([args.themes])[0]
        custom_theme_tensor = torch.as_tensor(custom_theme_one_hot).reshape(1, -1).to(device=device, dtype=torch.float32)
        scaled_r = scale_ratings(torch.tensor([args.target_rating]))[0]
        custom_rating_tensor = scaled_r.reshape(1).to(device=device, dtype=torch.float32)
        print(f"Using fixed custom themes: {args.themes}, target rating: {args.target_rating}", flush=True)
    else:
        print("Using randomly sampled target themes and ratings from dataset.", flush=True)

    buffer_rows = []
    total_saved_positions = 0
    executor = ThreadPoolExecutor(max_workers=n_jobs)

    try:
        for batch_start in range(0, args.n_positions, args.batch_size):
            batch_end = min(batch_start + args.batch_size, args.n_positions)
            curr_batch_size = batch_end - batch_start
            print(f"\n=== Processing Position Batch: Positions {batch_start + 1} to {batch_end} (Batch Size: {curr_batch_size}) ===", flush=True)

            batch_initial_tokens = initial_tokens.to(device=device).unsqueeze(0).repeat(curr_batch_size, 1)

            if config.use_context:
                if use_custom_themes:
                    batch_base_themes = [args.themes] * curr_batch_size
                    batch_base_ratings = [args.target_rating] * curr_batch_size
                    batch_theme_tensor = custom_theme_tensor.repeat(curr_batch_size, 1)
                    batch_rating_tensor = custom_rating_tensor.repeat(curr_batch_size)
                else:
                    batch_base_themes = []
                    batch_base_ratings = []
                    theme_tensors_list = []
                    rating_tensors_list = []
                    for _ in range(curr_batch_size):
                        rand_idx = random.randint(0, len(dataset) - 1)
                        sampled_item = dataset[rand_idx]
                        sampled_theme = torch.as_tensor(sampled_item[2])
                        sampled_rating = torch.as_tensor(sampled_item[3])
                        
                        b_theme = theme_preprocessor.inverse_transform(sampled_theme.reshape(1, -1).numpy())[0]
                        b_rating = unscale_ratings(sampled_rating).item()
                        batch_base_themes.append(b_theme)
                        batch_base_ratings.append(b_rating)
                        theme_tensors_list.append(sampled_theme.reshape(1, -1))
                        rating_tensors_list.append(sampled_rating.reshape(1))
                    
                    batch_theme_tensor = torch.cat(theme_tensors_list, dim=0).to(device=device, dtype=torch.float32)
                    batch_rating_tensor = torch.cat(rating_tensors_list, dim=0).to(device=device, dtype=torch.float32)
            else:
                batch_base_themes = [None] * curr_batch_size
                batch_base_ratings = [None] * curr_batch_size
                batch_theme_tensor = None
                batch_rating_tensor = None

            print(f"  Running GPU sampling from step {start_step} -> 0 for batch of {curr_batch_size} sequences...", flush=True)
            start_time = perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                sampled_tokens = sample_partial(
                    model,
                    batch_initial_tokens,
                    start_step_idx=start_step,
                    end_step_idx=0,
                    steps=args.steps,
                    temperature=args.temperature,
                    theme_tokens=batch_theme_tensor,
                    ratings=batch_rating_tensor
                )
            sampling_time = perf_counter() - start_time
            print(f"    GPU Sampling completed in {sampling_time:.3f}s", flush=True)

            sampled_tokens_cpu = sampled_tokens.cpu()

            eval_tasks = []
            for i in range(curr_batch_size):
                pos_idx = batch_start + i
                leaf_info = {
                    "node_id": str(pos_idx),
                    "tokens": sampled_tokens_cpu[i],
                }
                base_th = batch_base_themes[i]
                eval_tasks.append((pos_idx, base_th, batch_base_ratings[i], leaf_info))

            print(f"  Evaluating {len(eval_tasks)} final positions with Stockfish engine pool...", flush=True)
            eval_futures = [
                (pos_idx, base_th, base_rat, executor.submit(evaluate_leaf_puzzle, leaf_info, base_th, engine_pool, config))
                for pos_idx, base_th, base_rat, leaf_info in eval_tasks
            ]

            for pos_idx, base_th, base_rat, fut in eval_futures:
                leaf_eval = fut.result()
                row = {
                    "position_id": pos_idx,
                    "initial_fen": args.initial_fen,
                    "target_best_move": args.best_move,
                    "target_themes": str(base_th) if base_th is not None else None,
                    "target_rating": base_rat,
                    "fen": leaf_eval["fen"],
                    "is_legal": leaf_eval["is_legal"],
                    "is_puzzle": leaf_eval["is_puzzle"],
                    "best_move": leaf_eval["best_move"],
                    "main_line": leaf_eval["main_line"],
                    "actual_themes": str(leaf_eval["actual_themes"]),
                    "themes_match": leaf_eval["themes_match"],
                    "counter_intuitive_value": leaf_eval["counter_intuitive_value"],
                }
                buffer_rows.append(row)

            # Flush completed batch to CSV
            flush_rows_to_csv(buffer_rows, output_path)
            total_saved_positions += len(buffer_rows)
            print(f"  --> Appended {len(buffer_rows)} positions from batch to CSV (Total saved: {total_saved_positions})", flush=True)
            buffer_rows.clear()

    finally:
        while not engine_pool.empty():
            eng = engine_pool.get()
            eng.quit()
        executor.shutdown()

    if buffer_rows:
        flush_rows_to_csv(buffer_rows, output_path)
        total_saved_positions += len(buffer_rows)
        buffer_rows.clear()

    print(f"\nCompleted run! Total {total_saved_positions} position rows written to: {output_path}")


if __name__ == "__main__":
    main()
