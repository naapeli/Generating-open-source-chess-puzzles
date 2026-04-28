import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
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
from tokenization.tokenization import tokens_to_fen, tokens_to_move
from metrics.themes import legal , get_unique_puzzle_from_fen, counter_intuitive
from metrics.diversity_filtering import ReplayBuffer
from metrics.rewards import good_piece_counts, inter_batch_distances, intra_batch_distances


def log_rewards(components: dict[str, float], rewards, step: int):
    for key, value in components.items():
        writer.add_scalar(f"Components/{key}", value, step)
    writer.add_scalar("Reward", rewards.float().mean().item(), step)

def log_metrics(loss, grad_norm, kl_divergence, clips, step):
    writer.add_scalar("Loss/Loss", loss, step)
    writer.add_scalar("Loss/Grad Norm", grad_norm, step)
    writer.add_scalar("Loss/KL divergence", kl_divergence, step)
    writer.add_scalar("Loss/Clips", clips, step)

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

def save_board(fen, tag):
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

def get_reward(x_t, entropy, config, step):
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
    # rating_penalty = torch.zeros(batch_size, dtype=torch.float32)

    generations = []
    for tokens in x_t:
        try:
            generations.append((tokens_to_fen(tokens[:config.fen_length]), tokens_to_move(tokens[config.fen_length:])))
        except:
            generations.append((None, None))
    
    try:
        results = list(Parallel(n_jobs=cpu_count)(delayed(get_stockfish_data)(fen, move) for fen, move in generations))
        puzzles = [r[0] for r in results]
        best_moves = [r[1] for r in results]
        found_cp_losses = [r[2] for r in results]
        pvs = [r[3] for r in results]
    except TimeoutError:
        print("Stockfish timed out")
        return torch.zeros(batch_size, dtype=torch.float32)
    
    valid_indices = []

    batch_fens = [fen for fen, move in generations]
    batch_pvs = pvs

    for i, (fen, move) in enumerate(generations):
        if fen is None or not legal(fen):
            continue
        legal_position[i] = 1

        piece_counts[i] = good_piece_counts(fen)
        
        move_matches[i] = (move == best_moves[i])
        if found_cp_losses[i] is not None:
            cp_losses[i] = found_cp_losses[i]

        counter_intuitive_solution[i] = counter_intuitive(fen, engine)
        
        sampled_fens, sampled_pvs, _, _ = buffer.sample(2000)
        pv = pvs[i]
        intra_batch_fen_dist[i], intra_batch_pv_dist[i] = intra_batch_distances(fen, pv, batch_fens, batch_pvs, i)
        inter_batch_fen_dist[i], inter_batch_pv_dist[i] = inter_batch_distances(fen, pv, sampled_fens, sampled_pvs)

        puzzle = puzzles[i]
        if puzzle is None:
            continue
        unique_solution[i] = 1

        valid_indices.append(i)

    is_valid = torch.zeros(batch_size, dtype=torch.bool)
    is_valid[valid_indices] = True
    valid_mask = is_valid

    intra_distances = intra_batch_fen_dist * intra_batch_pv_dist
    inter_distances = inter_batch_fen_dist * inter_batch_pv_dist
    all_distances = intra_distances * inter_distances

    unique_and_counter_intuitive = unique_solution & counter_intuitive_solution
    
    good_intra_fen = intra_batch_fen_dist >= 6
    good_intra_pv = intra_batch_pv_dist >= 1
    good_inter_fen = inter_batch_fen_dist >= 6
    good_inter_pv = inter_batch_pv_dist >= 1
    pass_diversity_filtering = good_intra_fen & good_intra_pv & good_inter_fen & piece_counts & (entropy > 0.6)

    components = {
        "legal_rate": legal_position.float().mean().item(),
        "uniqueness_rate": unique_solution.float().mean().item(),
        "counter_intuitive_rate": counter_intuitive_solution.float().mean().item(),
        "counter_intuitive_rate_given_unique": counter_intuitive_solution[valid_mask].float().mean().item() if valid_mask.any() else 0,
        "unique_and_counter_intuitive": unique_and_counter_intuitive.float().mean().item(),
        "entropy": entropy.float().mean().item(),
        "piece_counts": piece_counts[legal_position].float().mean().item() if legal_position.any() else 0,
        # "themes_match_rate": themes_match[valid_mask].float().mean().item() if valid_mask.any() and config.use_context else 0,
        "dist_inter_fen": inter_batch_fen_dist[legal_position].float().mean().item() if legal_position.any() else 0,
        "dist_intra_fen": intra_batch_fen_dist[legal_position].float().mean().item() if legal_position.any() else 0,
        "dist_inter_pv": inter_batch_pv_dist[legal_position].float().mean().item() if legal_position.any() else 0,
        "dist_intra_pv": intra_batch_pv_dist[legal_position].float().mean().item() if legal_position.any() else 0,
        "intra_dist": intra_distances[legal_position].float().mean().item() if legal_position.any() else 0,
        "inter_dist": inter_distances[legal_position].float().mean().item() if legal_position.any() else 0,
        "all_dist": all_distances[legal_position].float().mean().item() if legal_position.any() else 0,
        # "rating_abs_diff": (-1000 * rating_penalty[valid_mask]).float().mean().item() if valid_mask.any() and config.use_context else 0,
        "pass_diversity_filtering": pass_diversity_filtering[legal_position].float().mean().item() if legal_position.any() else 0,
        "move_match_rate": move_matches[legal_position].float().mean().item() if legal_position.any() and config.predict_moves else 0,
        "cp_loss": cp_losses[valid_mask & ~torch.isnan(cp_losses)].mean().item() if (valid_mask & ~torch.isnan(cp_losses)).any() and config.predict_moves else 0,
    }

    rewards = torch.zeros(batch_size, dtype=torch.float32)
    rewards = torch.where(legal_position & pass_diversity_filtering & unique_and_counter_intuitive, 1, rewards)
    rewards = torch.where(~legal_position, -2, rewards)
    rewards = rewards.to(torch.float32)

    log_rewards(components, rewards, step)

    fens = [generation[0] for generation in generations]
    index = torch.argmax(rewards).item()
    save_board(fens[index], "Generations")
    for index, reward in enumerate(rewards):
        if not (unique_solution[index] & counter_intuitive_solution[index] & pass_diversity_filtering[index]):
            continue
        save_board(fens[index], "Puzzles")
    
    return rewards.to(x_t.device, dtype=torch.float32)

def seq_log_prob(log_probs, x_t, x_s, MASK_ID):
    mask = ((x_t == MASK_ID) & (x_s != MASK_ID)).float()
    log_probs = torch.cat([log_probs, torch.zeros((log_probs.shape[0], log_probs.shape[1], 1), device=log_probs.device)], dim=2)
    x_s_log_probs = log_probs.gather(dim=-1, index=x_s.unsqueeze(-1)).squeeze(-1)
    token_log_probs = x_s_log_probs * mask
    return token_log_probs.sum(dim=-1)

def kl_divergence(model_log_probs, ref_log_probs, x_t, x_s, MASK_ID):
    mask = ((x_t == MASK_ID) & (x_s != MASK_ID)).float()
    kl_div_vocab = F.kl_div(input=ref_log_probs, target=model_log_probs, log_target=True, reduction="none").sum(dim=2)
    token_kl = kl_div_vocab * mask
    return token_kl.sum(dim=1)  #  / (mask.sum(dim=1) + 1e-8)  # I think we should not normalize by the amount of unmasked tokens

def entropy(log_probs, x_t, x_s, MASK_ID):
    mask = ((x_t == MASK_ID) & (x_s != MASK_ID)).float()
    entropy = -(torch.exp(log_probs) * log_probs).sum(dim=2) * mask
    return entropy.sum(dim=1)

def train_ddpo(model, ref_model, optimizer, config, device, args, step):
    model.eval()
    with torch.no_grad():
        x_t = torch.full((args.batch_size, model.seq_length), config.mask_token, dtype=torch.long, device=device)
        
        trajectories = [] # Tuples of (x_t, x_s, p_mask, old_log_probs)

        T_grid = torch.linspace(0, 1, args.steps + 1, device=device)
        total_entropy = torch.zeros(args.batch_size, device=device)
        total_kl_divergence = torch.zeros(args.batch_size, device=device)

        for i in range(args.steps, 0, -1):
            t = T_grid[i]
            s = T_grid[i - 1]
            alpha_t = config.masking_schedule(t)
            alpha_s = config.masking_schedule(s)

            logits = model(x_t, None, None)
            model_log_probs = F.log_softmax(logits, dim=2)
            model_probs = torch.exp(model_log_probs)

            ref_logits = ref_model(x_t, None, None)
            ref_log_probs = F.log_softmax(ref_logits, dim=2)

            p_unmask = (alpha_s - alpha_t) / (1.0 - alpha_t + 1e-13)
            p_mask = (1.0 - alpha_s) / (1.0 - alpha_t + 1e-13)
            p_mask_tensor = p_mask.view(1, 1, 1).expand(args.batch_size, model.seq_length, 1)
            probs = torch.cat([model_probs * p_unmask, p_mask_tensor], dim=2)

            dist = torch.distributions.Categorical(probs)
            sampled_tokens = dist.sample()
            
            mask = (x_t == config.mask_token)
            x_s = torch.where(mask, sampled_tokens, x_t)

            step_entropy = entropy(model_log_probs, x_t, x_s, config.mask_token) / model.seq_length
            total_entropy += step_entropy

            step_kl = kl_divergence(model_log_probs, ref_log_probs, x_t, x_s, config.mask_token) / model.seq_length
            total_kl_divergence += step_kl
            
            old_log_probs = seq_log_prob(model_log_probs, x_t, x_s, config.mask_token)

            trajectories.append({
                "x_t": x_t,
                "x_s": x_s,
                "p_mask": p_mask,
                "old_log_probs": old_log_probs,
                "step_kl": step_kl,
                "step_entropy": step_entropy
            })
            
            x_t = x_s

        rewards = torch.zeros((args.batch_size, args.steps), dtype=torch.float32, device=device)
        rewards[:, -1] = get_reward(x_t, total_entropy.cpu(), config, step)  # (batch_size, steps)
        rewards[:, -1] += args.entropy_coef * total_entropy
        
        kl_divergences = torch.stack([t["step_kl"] for t in trajectories], dim=1)  # (batch_size, steps) (T => 0)
        entropies = torch.stack([t["step_entropy"] for t in trajectories], dim=1)  # (batch_size, steps) (T => 0)
        rewards -= args.kl_coef * kl_divergences# - args.entropy_coef * entropies
        
        returns = torch.zeros_like(rewards)
        running_return = torch.zeros(args.batch_size, dtype=torch.float32, device=device)

        for t in reversed(range(args.steps)):
            running_return = rewards[:, t] + args.gamma * running_return
            returns[:, t] = running_return

        advantages = (returns - returns.mean())  #  / (returns.std() + 1e-8)
        advantages = advantages.T.reshape(-1)
    
    x_t = torch.cat([t["x_t"] for t in trajectories], dim=0)
    x_s = torch.cat([t["x_s"] for t in trajectories], dim=0)
    p_mask = torch.cat([t["p_mask"].view(1).expand(args.batch_size) for t in trajectories], dim=0)
    old_log_probs = torch.cat([t["old_log_probs"] for t in trajectories], dim=0)

    dataset = TensorDataset(x_t, x_s, p_mask, old_log_probs, advantages)
    dataloader = DataLoader(dataset, batch_size=args.ppo_minibatch_size, shuffle=True)

    model.train()
    for epoch in range(args.ppo_epochs):
        total_loss = 0
        n_clips = 0
        total_norm = 0

        optimizer.zero_grad()

        # accumulate the gradients of the whole dataset
        for x_t, x_s, p_mask, old_log_probs, advantages in dataloader:
            logits = model(x_t, None, None)
            model_log_probs = F.log_softmax(logits, dim=2)
            model_probs = torch.exp(model_log_probs)
            
            p_unmask = (1 - p_mask).view(-1, 1, 1)
            p_mask_tensor = p_mask.view(-1, 1, 1).expand(-1, model.seq_length, 1)
            probs = torch.cat([model_probs * p_unmask, p_mask_tensor], dim=2)
            
            new_log_probs = seq_log_prob(model_log_probs, x_t, x_s, config.mask_token)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - args.eps, 1.0 + args.eps) * advantages
            loss = -torch.min(surr1, surr2).sum() / (args.steps * args.batch_size)

            loss.backward()
            
            total_loss += loss.item()
            n_clips += (surr2 < surr1).float().mean().item()

        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()

        optimizer.step()

        log_metrics(total_loss / len(dataloader), total_norm / len(dataloader), total_kl_divergence.mean().item(), n_clips / len(dataloader), step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--reference_path", type=str, required=True)
    parser.add_argument("--n_generations", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ppo_minibatch_size", type=int, default=4096)
    # parser.add_argument("--save_period", type=int, default=2000)
    # parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--kl_coef", type=float, default=0.03)
    parser.add_argument("--entropy_coef", type=float, default=0.03)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--ppo_epochs", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=1.0)
    args = parser.parse_args()

    cpu_count = int(os.environ.get("SLURM_CPUS_PER_TASK")) * int(os.environ.get("SLURM_NTASKS"))

    base_path = Path("./src")
    path = base_path / "runs" / "rl" / "ddpo" / args.run_name
    path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(path)

    engine = SimpleEngine.popen_uci(base_path / ".." / "Stockfish" / "src" / "stockfish")
    capacity = 200_000
    buffer = ReplayBuffer(capacity, base_path / "dataset" / "rl")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    checkpoint = torch.load(args.reference_path, map_location="cpu", weights_only=False)

    config = checkpoint["config"]
    model = MaskedDiffusion(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device=device)

    reference_model = MaskedDiffusion(checkpoint["config"])
    reference_model.load_state_dict(checkpoint["model"])
    reference_model.to(device=device)
    reference_model.eval()
    reference_model.requires_grad_(False)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    for step in range(args.n_generations):
        train_ddpo(model, reference_model, optimizer, config, device, args, step)
