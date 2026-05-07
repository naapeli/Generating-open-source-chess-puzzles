import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.tensorboard import SummaryWriter
import chess
from chess.engine import SimpleEngine, Limit
from chess import svg
import cairosvg
from PIL import Image, ImageDraw
import io
import numpy as np

from pathlib import Path
import argparse
from joblib import Parallel, delayed
import os

from MaskedDiffusion.model import MaskedDiffusion
from tokenization.tokenization import tokens_to_fen, tokens_to_move, tokenize_fen, tokenize_move
from metrics.themes import legal, get_unique_puzzle_from_fen, counter_intuitive
from metrics.diversity_filtering import ReplayBuffer
from metrics.rewards import good_piece_counts, inter_batch_distances, intra_batch_distances
from MaskingSchedule.MaskingSchedule import string_to_schedule

from rl.espo import compute_elbo, kl_estimate, entropy

def log_metrics(total_loss, grad_norm, kl_divergence, clips, step, lr, reward=None):
    writer.add_scalar("Loss/Total Loss", total_loss, step)
    writer.add_scalar("Loss/RL Grad Norm", grad_norm, step)
    writer.add_scalar("Loss/KL divergence", kl_divergence, step)
    writer.add_scalar("Loss/Clips", clips, step)
    writer.add_scalar("Loss/learning_rate", lr, step)
    if reward is not None:
        writer.add_scalar("Reward", reward, step)

def get_stockfish_data(fen, model_move):
    if fen is None: return None, None, None, None
    if not legal(fen): return None, None, None, None
    with SimpleEngine.popen_uci(base_path / ".." / "Stockfish" / "src" / "stockfish") as stockfish:
        board = chess.Board(fen)
        limit = Limit(depth=15, time=10, nodes=8_000_000)
        
        analysis = stockfish.analyse(board, limit=limit)
        best_move = analysis["pv"][0] if "pv" in analysis else None
        pv_string = " ".join([move.uci() for move in analysis["pv"]]) if "pv" in analysis else ""
        player_to_move = board.turn
        best_score = analysis["score"].pov(player_to_move) if "score" in analysis else None
        
        puzzle = get_unique_puzzle_from_fen(fen, stockfish)
        
        cp_loss = None
        if model_move and best_score is not None:
            try:
                move = chess.Move.from_uci(model_move)
                if move in board.legal_moves:
                    if best_move and move == best_move:
                        cp_loss = 0
                    else:
                        board.push(move)
                        model_analysis = stockfish.analyse(board, limit=limit)
                        model_score = model_analysis["score"].pov(player_to_move)
                        board.pop()
                        
                        if not best_score.is_mate() and not model_score.is_mate():
                            cp_loss = max(0, best_score.score() - model_score.score())
            except:
                pass

        if best_move is None:
            return puzzle, None, cp_loss, pv_string
    return puzzle, best_move.uci(), cp_loss, pv_string

def save_board(fen, tag, step):
    try:
        board = chess.Board(fen)
        svg_data = svg.board(board, size=300)
        png_data = cairosvg.svg2png(bytestring=svg_data.encode("utf-8"))
        board_img = Image.open(io.BytesIO(png_data)).convert("RGB")
        text_height = 50
        info_pane = Image.new("RGB", (board_img.width, text_height), (255, 255, 255))
        draw = ImageDraw.Draw(info_pane)
        draw.text((10, 10), fen, fill=(0, 0, 0))
        combined_img = np.vstack((np.array(board_img), np.array(info_pane)))
        
        writer.add_image(tag, combined_img, step, dataformats="HWC")
        writer.add_text(f"{tag}/fen", fen, step)
    except Exception:
        pass

def get_reward(x_t, entropy_vals, config, step):
    x_t_cpu = x_t.cpu()
    batch_size = len(x_t_cpu)

    legal_position = torch.zeros(batch_size, dtype=bool)
    unique_solution = torch.zeros(batch_size, dtype=bool)
    counter_intuitive_solution = torch.zeros(batch_size, dtype=bool)
    piece_counts = torch.zeros(batch_size, dtype=bool)
    inter_batch_fen_dist = torch.zeros(batch_size, dtype=torch.float32)
    intra_batch_fen_dist = torch.zeros(batch_size, dtype=torch.float32)
    inter_batch_pv_dist = torch.zeros(batch_size, dtype=torch.float32)
    intra_batch_pv_dist = torch.zeros(batch_size, dtype=torch.float32)
    move_matches = torch.zeros(batch_size, dtype=bool)
    cp_losses = torch.full((batch_size,), float("nan"), dtype=torch.float32)

    generations = []
    for tokens in x_t:
        try:
            generations.append((tokens_to_fen(tokens[:config.fen_length]), tokens_to_move(tokens[config.fen_length:])))
        except:
            generations.append((None, None))
    
    try:
        results = list(Parallel(n_jobs=cpu_count)(delayed(get_stockfish_data)(fen, move) for fen, move in generations))
    except Exception as e:
        print(e)
        return torch.zeros(batch_size, dtype=torch.float32, device=x_t.device)
        
    puzzles = [r[0] for r in results]
    best_moves = [r[1] for r in results]
    found_cp_losses = [r[2] for r in results]
    pvs = [r[3] for r in results]
    
    valid_indices = []

    batch_fens = [fen for fen, move in generations]
    batch_pvs = pvs

    sampled_fens, sampled_pvs, _, _ = buffer.sample(2000)

    for i, (fen, move) in enumerate(generations):
        if fen is None or not legal(fen):
            continue
        legal_position[i] = 1

        piece_counts[i] = good_piece_counts(fen)
        
        move_matches[i] = (move == best_moves[i])
        if found_cp_losses[i] is not None:
            cp_losses[i] = found_cp_losses[i]

        with SimpleEngine.popen_uci(base_path / ".." / "Stockfish" / "src" / "stockfish") as engine:
            counter_intuitive_solution[i] = counter_intuitive(fen, engine)
        
        pv = pvs[i]
        intra_batch_fen_dist[i], intra_batch_pv_dist[i] = intra_batch_distances(fen, pv, batch_fens, batch_pvs, i)
        inter_batch_fen_dist[i], inter_batch_pv_dist[i] = inter_batch_distances(fen, pv, sampled_fens, sampled_pvs)

        puzzle = puzzles[i]
        if puzzle is None:
            continue
        unique_solution[i] = 1

        # if the position returns a high reward, add it to the buffer
        good_distances = (intra_batch_fen_dist[i] >= 6) and (intra_batch_pv_dist[i] >= 1) and (inter_batch_fen_dist[i] >= 6)
        if unique_solution[i] and counter_intuitive_solution[i] and piece_counts[i] and good_distances:
            buffer.add(fen, pv, [], -1)

        valid_indices.append(i)

    is_valid = torch.zeros(batch_size, dtype=torch.bool)
    is_valid[valid_indices] = True

    unique_and_counter_intuitive = unique_solution & counter_intuitive_solution
    
    good_intra_fen = intra_batch_fen_dist >= 6
    good_intra_pv = intra_batch_pv_dist >= 1
    good_inter_fen = inter_batch_fen_dist >= 6
    good_inter_pv = inter_batch_pv_dist >= 1

    intra_distances = good_intra_fen & good_intra_pv
    inter_distances = good_inter_fen & good_inter_pv
    all_distances = intra_distances & inter_distances

    pass_diversity_filtering = good_intra_fen & good_inter_fen & good_intra_pv & piece_counts   # & (entropy_vals > 0.6)

    components = {
        "legal_rate": legal_position.float().mean().item(),
        "uniqueness_rate": unique_solution.float().mean().item(),
        "counter_intuitive_rate": counter_intuitive_solution.float().mean().item(),
        "counter_intuitive_rate_given_unique": counter_intuitive_solution[is_valid].float().mean().item() if is_valid.any() else 0,
        "unique_and_counter_intuitive": unique_and_counter_intuitive.float().mean().item(),
        "entropy": entropy_vals.float().mean().item(),
        "piece_counts": piece_counts[legal_position].float().mean().item() if legal_position.any() else 0,
        "dist_inter_fen": inter_batch_fen_dist[legal_position].float().mean().item() if legal_position.any() else 0,
        "dist_intra_fen": intra_batch_fen_dist[legal_position].float().mean().item() if legal_position.any() else 0,
        "dist_inter_pv": inter_batch_pv_dist[legal_position].float().mean().item() if legal_position.any() else 0,
        "dist_intra_pv": intra_batch_pv_dist[legal_position].float().mean().item() if legal_position.any() else 0,
        "intra_dist": intra_distances[legal_position].float().mean().item() if legal_position.any() else 0,
        "inter_dist": inter_distances[legal_position].float().mean().item() if legal_position.any() else 0,
        "all_dist": all_distances[legal_position].float().mean().item() if legal_position.any() else 0,
        "pass_diversity_filtering": pass_diversity_filtering[legal_position].float().mean().item() if legal_position.any() else 0,
        "move_match_rate": move_matches[legal_position].float().mean().item() if legal_position.any() and config.predict_moves else 0,
        "cp_loss": cp_losses[is_valid & ~torch.isnan(cp_losses)].mean().item() if (is_valid & ~torch.isnan(cp_losses)).any() and config.predict_moves else 0,
    }

    rewards = torch.zeros(batch_size, dtype=torch.float32)
    rewards = torch.where(legal_position & pass_diversity_filtering & unique_solution, 0.1, rewards)
    rewards = torch.where(legal_position & pass_diversity_filtering & unique_and_counter_intuitive, 1.0, rewards)
    rewards = torch.where(~legal_position, -2.0, rewards)

    rewards = rewards.to(torch.float32)

    for key, value in components.items():
        writer.add_scalar(f"Components/{key}", value, step)
    writer.add_scalar("Reward", rewards.float().mean().item(), step)

    log_rewards = rewards.clone()
    index = torch.argmax(log_rewards).item()
    save_board(batch_fens[index], "Generations", step)
    for index, reward in enumerate(log_rewards):
        if not (unique_solution[index] & counter_intuitive_solution[index] & pass_diversity_filtering[index]):
            continue
        save_board(batch_fens[index], "Puzzles", step)
    
    return rewards.to(x_t.device, dtype=torch.float32)

def train_espo(model, reference_model, optimizer, scheduler, config, device, args, step):
    model.eval()
    
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        step_fens = model.sample(None, None, steps=args.steps, batch_size=args.batch_size, temperature=args.temperature, generate_move_last=False)
    
    if args.n_arteficial > 0:
        sampled_fens, sampled_pvs, _, _ = buffer.sample(args.n_arteficial)
        x_0_art_list = []
        for fen, pv in zip(sampled_fens, sampled_pvs):
            fen_tokens = tokenize_fen(fen)
            move_str = pv.split(" ")[0] if " " in pv else pv
            move_tokens = tokenize_move(move_str)
            x_0_art_list.append(fen_tokens + move_tokens)
        x_0_art = torch.tensor(x_0_art_list, dtype=torch.long, device=device)
        
        combined_fens = torch.cat([step_fens, x_0_art], dim=0)
    else:
        combined_fens = step_fens

    with torch.no_grad():
        reference_elbo, mask, t = compute_elbo(reference_model, combined_fens, themes=None, ratings=None, return_mask=True)
        old_elbo = compute_elbo(model, combined_fens, themes=None, ratings=None, mask=mask)

    old_elbo_gen = old_elbo[:args.batch_size]
    entropy_vals = entropy(old_elbo_gen.cpu(), step_fens.shape[1])
    
    rewards_gen = get_reward(step_fens, entropy_vals, config, step)

    if args.n_arteficial > 0:
        rewards_art = torch.full((args.n_arteficial,), 1.0, device=device)
        rewards = torch.cat([rewards_gen, rewards_art], dim=0)
    else:
        rewards = rewards_gen

    model.train()
    total_loss = 0
    total_norm = 0
    total_kl = 0
    total_clips = 0

    for substep in range(args.n_gradient_updates_per_generation):
        optimizer.zero_grad()
        
        elbo = compute_elbo(model, combined_fens, themes=None, ratings=None, mask=mask, return_mask=False)
        
        sequence_length = combined_fens.shape[1]
        elbo_scaled = elbo / sequence_length
        reference_elbos_scaled = reference_elbo / sequence_length
        old_elbos_scaled = old_elbo / sequence_length

        # kl = sequence_length * kl_estimate(elbo_scaled, reference_elbos_scaled)
        kl = kl_estimate(elbo, reference_elbo)

        # rewards = rewards - args.beta * kl
        rewards = rewards.detach()
        # kl = kl.mean()
        kl = kl.mean() / sequence_length  # TODO
        
        rho = torch.exp(elbo_scaled - old_elbos_scaled)
        
        N = rewards.shape[0]
        # mean_other_rewards = (rewards.sum() - rewards) / (N - 1)
        # advantages = (rewards - mean_other_rewards).to(device)
        advantages = ((rewards - rewards.mean()) / (rewards.std() + 1e-6)).to(device)  # TODO
        
        coef_1 = rho * advantages
        coef_2 = torch.clamp(rho, 1 - args.eps, 1 + args.eps) * advantages
        loss = -torch.minimum(coef_1, coef_2).mean()
        
        is_clipped = (coef_2 < coef_1).flatten()
        
        loss = loss + args.beta * kl
        
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        total_norm += norm.item()
        total_kl += kl.item()
        total_clips += is_clipped.float().mean().item()

    log_metrics(
        total_loss / args.n_gradient_updates_per_generation,
        total_norm / args.n_gradient_updates_per_generation,
        total_kl / args.n_gradient_updates_per_generation,
        total_clips / args.n_gradient_updates_per_generation,
        step,
        optimizer.param_groups[0]["lr"],
        rewards_gen.float().mean().item()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--reference_path", type=str, required=True)
    parser.add_argument("--n_generations", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_arteficial", type=int, default=4)
    parser.add_argument("--n_gradient_updates_per_generation", type=int, default=1)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument("--eps", type=float, default=0.2)
    args = parser.parse_args()

    cpu_count = int(os.environ.get("SLURM_CPUS_PER_TASK", "1")) * int(os.environ.get("SLURM_NTASKS", "1"))

    base_path = Path("./src")
    path = base_path / "runs" / "rl" / "espov3" / args.run_name
    path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(path)

    capacity = 200_000
    buffer = ReplayBuffer(capacity, base_path / "dataset" / "rl")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    checkpoint = torch.load(args.reference_path, map_location="cpu", weights_only=False)

    config = checkpoint["config"]
    config.schedule = "linear"
    config.masking_schedule = string_to_schedule(config.schedule)
    model = MaskedDiffusion(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device=device)

    reference_model = MaskedDiffusion(checkpoint["config"])
    reference_model.load_state_dict(checkpoint["model"])
    reference_model.to(device=device)
    reference_model.eval()
    reference_model.requires_grad_(False)

    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.1)
    scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=3)

    for step in range(args.n_generations):
        train_espo(model, reference_model, optimizer, scheduler, config, device, args, step)
