import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
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
from concurrent.futures import ThreadPoolExecutor
import os

from MaskedDiffusion.model import MaskedDiffusion
from tokenization.tokenization import theme_preprocessor, scale_ratings, unscale_ratings, tokens_to_fen, tokens_to_move, tokenize_fen, tokenize_move
from metrics.themes import legal , get_unique_puzzle_from_fen, counter_intuitive
from metrics.diversity_filtering import ReplayBuffer
from metrics.rewards import good_piece_counts, inter_batch_distances, intra_batch_distances
from MaskingSchedule.MaskingSchedule import string_to_schedule
from rl.espo import generate_random_themes


def log_rewards(components: dict[str, float], rewards, step: int):
    for key, value in components.items():
        writer.add_scalar(f"Components/{key}", value, step)

def log_metrics(total_loss, grad_norm, kl_divergence, clips, step, lr, reward=None, entropy=None):
    writer.add_scalar("Loss/Total Loss", total_loss, step)
    writer.add_scalar("Loss/RL Grad Norm", grad_norm, step)
    writer.add_scalar("Loss/KL divergence", kl_divergence, step)
    writer.add_scalar("Loss/Clips", clips, step)
    writer.add_scalar("Loss/learning_rate", lr, step)
    if reward is not None:
        writer.add_scalar("Reward", reward, step)
    if entropy is not None:
        writer.add_scalar("Loss/Entropy", entropy, step)

def get_stockfish_data(fen, model_move):
    if fen is None: return None, None, None, None, False, 0.0
    if not legal(fen): return None, None, None, None, False, 0.0
    with SimpleEngine.popen_uci(base_path / ".." / "Stockfish" / "src" / "stockfish") as stockfish:
        stockfish.configure({"Threads": 1, "Hash": 32})
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

        stockfish.configure({"Clear Hash": None})
        ci_sol, ci_val = counter_intuitive(fen, stockfish, return_value=True)

        if best_move is None:
            return puzzle, None, cp_loss, pv_string, ci_sol, ci_val
    return puzzle, best_move.uci(), cp_loss, pv_string, ci_sol, ci_val

def save_board(fen, tag, step, themes=None, rating=None):
    try:
        board = chess.Board(fen)
        svg_data = svg.board(board, size=300)
        png_data = cairosvg.svg2png(bytestring=svg_data.encode("utf-8"))
        board_img = Image.open(io.BytesIO(png_data)).convert("RGB")
        text_height = 80 if themes is not None else 50
        info_pane = Image.new("RGB", (board_img.width, text_height), (255, 255, 255))
        draw = ImageDraw.Draw(info_pane)
        text_content = f"{rating}\n{themes}\n{fen}" if themes is not None else fen
        draw.text((10, 10), text_content, fill=(0, 0, 0))
        combined_img = np.vstack((np.array(board_img), np.array(info_pane)))
        
        writer.add_image(tag, combined_img, step, dataformats="HWC")
        writer.add_text(f"{tag}/fen", text_content, step)
    except Exception:
        pass

def get_reward(x_t, entropy, config, step, themes_tokens=None, ratings=None):
    x_t_cpu = x_t.cpu()
    batch_size = len(x_t_cpu)

    legal_position = torch.zeros(batch_size, dtype=bool)
    unique_solution = torch.zeros(batch_size, dtype=bool)
    counter_intuitive_solution = torch.zeros(batch_size, dtype=bool)
    counter_intuitive_values = torch.zeros(batch_size, dtype=torch.float32)
    piece_counts = torch.zeros(batch_size, dtype=bool)
    inter_batch_fen_dist = torch.zeros(batch_size, dtype=torch.float32)
    intra_batch_fen_dist = torch.zeros(batch_size, dtype=torch.float32)
    inter_batch_pv_dist = torch.zeros(batch_size, dtype=torch.float32)
    intra_batch_pv_dist = torch.zeros(batch_size, dtype=torch.float32)
    move_matches = torch.zeros(batch_size, dtype=bool)
    cp_losses = torch.full((batch_size,), float("nan"), dtype=torch.float32)
    # rating_penalty = torch.zeros(batch_size, dtype=torch.float32)

    if config.use_context and themes_tokens is not None:
        themes = theme_preprocessor.inverse_transform(themes_tokens.cpu().numpy())
    else:
        themes = [None] * batch_size
    
    if config.use_context and ratings is not None:
        true_ratings = ratings.cpu().tolist()
    else:
        true_ratings = [None] * batch_size

    generations = []
    for tokens in x_t:
        try:
            generations.append((tokens_to_fen(tokens[:config.fen_length]), tokens_to_move(tokens[config.fen_length:])))
        except:
            generations.append((None, None))
    
    try:
        with ThreadPoolExecutor(max_workers=cpu_count) as executor:
            results = list(executor.map(lambda p: get_stockfish_data(p[0], p[1]), generations))
        puzzles = [r[0] for r in results]
        best_moves = [r[1] for r in results]
        found_cp_losses = [r[2] for r in results]
        pvs = [r[3] for r in results]
        ci_solutions = [r[4] for r in results]
        ci_values = [r[5] for r in results]
    except TimeoutError:
        print("Stockfish timed out")
        return torch.zeros(batch_size, dtype=torch.float32)
    
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

        counter_intuitive_solution[i] = ci_solutions[i]
        counter_intuitive_values[i] = ci_values[i]
        
        pv = pvs[i]
        intra_batch_fen_dist[i], intra_batch_pv_dist[i] = intra_batch_distances(fen, pv, batch_fens, batch_pvs, i)
        inter_batch_fen_dist[i], inter_batch_pv_dist[i] = inter_batch_distances(fen, pv, sampled_fens, sampled_pvs)

        puzzle = puzzles[i]
        if puzzle is None:
            continue
        unique_solution[i] = 1

        # if the position returns a high reward, add it to the buffer
        good_distances = (intra_batch_fen_dist[i] >= 6) and (intra_batch_pv_dist[i] >= 1) and (inter_batch_fen_dist[i] >= 6)   # and (inter_batch_pv_dist[i] >= 1)
        if unique_solution[i] and counter_intuitive_solution[i] and piece_counts[i] and good_distances:
            buffer_themes = themes[i] if themes[i] is not None else []
            buffer_rating = true_ratings[i] if true_ratings[i] is not None else -1
            buffer.add(fen, pv, buffer_themes, buffer_rating)

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

    pass_diversity_filtering = good_intra_fen & good_inter_fen & good_intra_pv & good_inter_pv & piece_counts & (entropy > 0.5)
    # pass_diversity_filtering = torch.ones(batch_size, dtype=bool)

    components = {
        "legal_rate": legal_position.float().mean().item(),
        "uniqueness_rate": unique_solution.float().mean().item(),
        "counter_intuitive_rate": counter_intuitive_solution.float().mean().item(),
        "counter_intuitive_values": counter_intuitive_values.mean().item(),
        "counter_intuitive_values_given_unique": counter_intuitive_values[is_valid].mean().item() if is_valid.any() else 0,
        "counter_intuitive_rate_given_unique": counter_intuitive_solution[is_valid].float().mean().item() if is_valid.any() else 0,
        "unique_and_counter_intuitive": unique_and_counter_intuitive.float().mean().item(),
        "piece_counts": piece_counts[legal_position].float().mean().item() if legal_position.any() else 0,
        # "themes_match_rate": themes_match[is_valid].float().mean().item() if is_valid.any() and config.use_context else 0,
        "dist_inter_fen": inter_batch_fen_dist[legal_position].float().mean().item() if legal_position.any() else 0,
        "dist_intra_fen": intra_batch_fen_dist[legal_position].float().mean().item() if legal_position.any() else 0,
        "dist_inter_pv": inter_batch_pv_dist[legal_position].float().mean().item() if legal_position.any() else 0,
        "dist_intra_pv": intra_batch_pv_dist[legal_position].float().mean().item() if legal_position.any() else 0,
        "intra_dist": intra_distances[legal_position].float().mean().item() if legal_position.any() else 0,
        "inter_dist": inter_distances[legal_position].float().mean().item() if legal_position.any() else 0,
        "all_dist": all_distances[legal_position].float().mean().item() if legal_position.any() else 0,
        # "rating_abs_diff": (-1000 * rating_penalty[is_valid]).float().mean().item() if is_valid.any() and config.use_context else 0,
        "pass_diversity_filtering": pass_diversity_filtering[legal_position].float().mean().item() if legal_position.any() else 0,
        "move_match_rate": move_matches[legal_position].float().mean().item() if legal_position.any() and config.predict_moves else 0,
        "cp_loss": cp_losses[is_valid & ~torch.isnan(cp_losses)].mean().item() if (is_valid & ~torch.isnan(cp_losses)).any() and config.predict_moves else 0,
    }

    rewards = torch.zeros(batch_size, dtype=torch.float32)
    # rewards = torch.where(legal_position & pass_diversity_filtering & unique_and_counter_intuitive, 1.0, rewards)
    # rewards = torch.where(legal_position & unique_solution, counter_intuitive_values, rewards)
    # rewards = torch.where(legal_position & unique_and_counter_intuitive, 1.0, rewards)
    rewards = torch.where(legal_position & pass_diversity_filtering & unique_solution, 10 * counter_intuitive_values, rewards)
    rewards = torch.where(~legal_position, -2.0, rewards)
    rewards = rewards.to(torch.float32)

    log_rewards(components, rewards, step)

    index = torch.argmax(rewards).item()
    save_board(batch_fens[index], "Generations", step, themes[index], true_ratings[index])
    index = torch.argmin(rewards).item()
    save_board(batch_fens[index], "Worst_Generations", step, themes[index], true_ratings[index])
    index = torch.randint(0, batch_size, (1,)).item()
    save_board(batch_fens[index], "Random_Generations", step, themes[index], true_ratings[index])
    for index, reward in enumerate(rewards):
        if not (unique_solution[index] & counter_intuitive_solution[index] & pass_diversity_filtering[index]):
            continue
        save_board(batch_fens[index], "Puzzles", step, themes[index], true_ratings[index])
    
    return rewards.to(x_t.device, dtype=torch.float32)

def kl_divergence(model_log_probs, ref_log_probs, x_t, p_unmask, MASK_ID):
    mask = (x_t == MASK_ID).float()
    kl_div_vocab = F.kl_div(input=ref_log_probs, target=model_log_probs, log_target=True, reduction="none").sum(dim=2)
    token_kl = kl_div_vocab * mask * p_unmask
    return token_kl.sum(dim=1)

def seq_log_prob(log_probs, x_t, x_s, p_unmask, MASK_ID):
    unmask_mask = ((x_t == MASK_ID) & (x_s != MASK_ID)).float()
    remain_masked_mask = ((x_t == MASK_ID) & (x_s == MASK_ID)).float()
    x_s_log_probs = log_probs.gather(dim=-1, index=x_s.clamp(max=log_probs.shape[-1]-1).unsqueeze(-1)).squeeze(-1)
    if isinstance(p_unmask, torch.Tensor) and p_unmask.dim() == 1:
        p_unmask = p_unmask.unsqueeze(1)
    log_p_unmask = torch.log(p_unmask + 1e-13)
    log_p_mask = torch.log(1.0 - p_unmask + 1e-13)
    token_log_probs = (x_s_log_probs + log_p_unmask) * unmask_mask + log_p_mask * remain_masked_mask
    return token_log_probs.sum(dim=-1)

def entropy(log_probs, x_t, p_unmask, MASK_ID):
    mask = (x_t == MASK_ID).float()
    entropy_vocab = -(torch.exp(log_probs) * log_probs).sum(dim=2)
    if isinstance(p_unmask, torch.Tensor) and p_unmask.dim() == 1:
        p_unmask = p_unmask.unsqueeze(1)
    token_entropy = entropy_vocab * mask * p_unmask
    return token_entropy.sum(dim=1)

def train_ddpo(model, ref_model, optimizer, scheduler, config, device, args, step):
    model.eval()
    with torch.no_grad():
        total_batch_size = args.batch_size * args.group_size
        x_t = torch.full((total_batch_size, model.seq_length), config.mask_token, dtype=torch.long, device=device)

        trajectories = [] # Tuples of (x_t, x_s, p_mask, old_log_probs)

        # Generate themes and ratings if use_context
        if config.use_context:
            themes, ratings = generate_random_themes(args.batch_size, lichess_distribution=args.lichess_distribution)
            ratings = ratings.to(device=device, dtype=torch.float32)
            themes_one_hot = torch.from_numpy(theme_preprocessor.transform(themes)).to(device=device, dtype=torch.float32)
            scaled_ratings = scale_ratings(ratings).to(device=device, dtype=torch.float32)
            
            # Repeat themes and ratings within each group
            themes_one_hot = themes_one_hot.repeat_interleave(args.group_size, dim=0)
            scaled_ratings = scaled_ratings.repeat_interleave(args.group_size, dim=0)
            ratings = ratings.repeat_interleave(args.group_size, dim=0)
        else:
            themes, ratings = None, None
            themes_one_hot = torch.zeros((total_batch_size, 1), device=device)
            scaled_ratings = torch.zeros((total_batch_size,), device=device)

        T_grid = torch.linspace(0, 1, args.steps + 1, device=device)
        total_entropy = torch.zeros(total_batch_size, device=device)
        total_kl_divergence = torch.zeros(total_batch_size, device=device)

        for i in range(args.steps, 0, -1):
            t = T_grid[i]
            s = T_grid[i - 1]
            alpha_t = config.masking_schedule(t)
            alpha_s = config.masking_schedule(s)
            if s == 0.0:
                alpha_s = torch.ones_like(alpha_s)

            logits = model(x_t, themes_one_hot if config.use_context else None, scaled_ratings if config.use_context else None)
            model_log_probs = F.log_softmax(logits / args.temperature, dim=2)
            model_probs = torch.exp(model_log_probs)

            ref_logits = ref_model(x_t, themes_one_hot if config.use_context else None, scaled_ratings if config.use_context else None)
            ref_log_probs = F.log_softmax(ref_logits / args.temperature, dim=2)

            log_p_unmask = torch.log(alpha_s - alpha_t) - torch.log(1.0 - alpha_t + 1e-13)
            p_unmask = torch.exp(log_p_unmask)
            log_p_mask = torch.log(1.0 - alpha_s) - torch.log(1.0 - alpha_t + 1e-13)
            log_p_mask_tensor = log_p_mask.view(1, 1, 1).expand(total_batch_size, model.seq_length, 1)
            log_probs = torch.cat([model_log_probs + log_p_unmask, log_p_mask_tensor], dim=2)
            
            dist = torch.distributions.Categorical(logits=log_probs)
            sampled_tokens = dist.sample()
            
            mask = (x_t == config.mask_token)
            x_s = torch.where(mask, sampled_tokens, x_t)

            step_entropy = entropy(model_log_probs, x_t, p_unmask, config.mask_token) / model.seq_length
            total_entropy += step_entropy

            step_kl = kl_divergence(model_log_probs, ref_log_probs, x_t, p_unmask, config.mask_token)
            total_kl_divergence += step_kl
            
            old_log_probs = seq_log_prob(model_log_probs, x_t, x_s, p_unmask, config.mask_token)

            trajectories.append({
                "x_t": x_t,
                "x_s": x_s,
                "old_log_probs": old_log_probs,
                "step_kl": step_kl,
                "step_entropy": step_entropy,
                "p_unmask": p_unmask.view(1).expand(total_batch_size)
            })
            
            x_t = x_s
            
            # Clean up sampling step tensors to prevent peak memory growth
            del logits, ref_logits, model_log_probs, model_probs, ref_log_probs
            del log_p_unmask, log_p_mask, log_p_mask_tensor, log_probs, dist, sampled_tokens, mask

        rewards = torch.zeros((total_batch_size, args.steps), dtype=torch.float32, device=device)
        rewards[:, -1] = get_reward(x_t, total_entropy.cpu(), config, step, themes_one_hot if config.use_context else None, ratings if config.use_context else None)
        
        kl_divergences = torch.stack([t["step_kl"] for t in trajectories], dim=1)  # (batch_size, steps) (T => 0)
        
        returns = torch.zeros_like(rewards)
        running_return = torch.zeros(total_batch_size, dtype=torch.float32, device=device)
        
        for t in reversed(range(args.steps)):
            running_return = rewards[:, t] + args.gamma * running_return
            returns[:, t] = running_return
            
        returns = returns + args.entropy_coef * total_entropy.unsqueeze(1) - args.kl_coef * kl_divergences
        
        reward_val = returns.float().mean().item()

        # Reshape returns to (num_groups, group_size, steps) to normalize over each group
        returns_grouped = returns.reshape(args.batch_size, args.group_size, args.steps)
        if args.group_size > 1:
            mean_other_returns = (returns_grouped.sum(dim=1, keepdim=True) - returns_grouped) / (args.group_size - 1)
        else:
            mean_other_returns = torch.zeros_like(returns_grouped)
        advantages_grouped = (returns_grouped - mean_other_returns) / (returns_grouped.std(dim=(1, 2), keepdim=True) + 1e-5)
        advantages = advantages_grouped.reshape(total_batch_size, args.steps).T.reshape(-1)
    
    x_t = torch.cat([t["x_t"] for t in trajectories], dim=0)
    x_s = torch.cat([t["x_s"] for t in trajectories], dim=0)
    old_log_probs = torch.cat([t["old_log_probs"] for t in trajectories], dim=0)
    p_unmasks = torch.cat([t["p_unmask"] for t in trajectories], dim=0)

    if config.use_context:
        themes_one_hot_all = themes_one_hot.repeat(args.steps, 1)
        scaled_ratings_all = scaled_ratings.repeat(args.steps)
    else:
        themes_one_hot_all = torch.zeros((x_t.shape[0], 1), device=device)
        scaled_ratings_all = torch.zeros((x_t.shape[0],), device=device)

    dataset = TensorDataset(x_t, x_s, old_log_probs, advantages, p_unmasks, themes_one_hot_all, scaled_ratings_all)
    
    # Explicitly release large intermediate tensors/lists that are now in dataset
    del x_t, x_s, old_log_probs, advantages, p_unmasks, themes_one_hot_all, scaled_ratings_all
    del trajectories, rewards, kl_divergences, returns, returns_grouped, advantages_grouped
    if args.group_size > 1:
        del mean_other_returns
    if config.use_context:
        del themes_one_hot, scaled_ratings, ratings
    else:
        del themes_one_hot, scaled_ratings
        
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    dataloader = DataLoader(dataset, batch_size=args.ppo_minibatch_size, shuffle=True)

    model.train()
    total_loss = 0
    n_clips = 0
    total_norm = 0
    for epoch in range(args.ppo_epochs):
        optimizer.zero_grad()
        for x_t_batch, x_s_batch, old_log_probs_batch, advantages_batch, p_unmask_batch, themes_batch, ratings_batch in dataloader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(
                    x_t_batch,
                    themes_batch if config.use_context else None,
                    ratings_batch if config.use_context else None,
                    checkpoint_activations=args.checkpoint_activations
                )
            model_log_probs = F.log_softmax(logits / args.temperature, dim=2)
            new_log_probs = seq_log_prob(model_log_probs, x_t_batch, x_s_batch, p_unmask_batch, config.mask_token)
            
            ratio = torch.exp(new_log_probs - old_log_probs_batch)
            surr1 = ratio * advantages_batch
            surr2 = torch.clamp(ratio, 1.0 - args.eps, 1.0 + args.eps) * advantages_batch
            
            loss_ppo = -torch.min(surr1, surr2).sum() / (args.steps * total_batch_size)
            loss_ppo.backward()
            
            total_loss += loss_ppo.item()
            n_clips += (surr2 < surr1).float().mean().item()
            
            # Explicitly delete minibatch tensors to prevent VRAM accumulation
            del logits, model_log_probs, new_log_probs, ratio, surr1, surr2, loss_ppo
            del x_t_batch, x_s_batch, old_log_probs_batch, advantages_batch, p_unmask_batch, themes_batch, ratings_batch
            
        # supervised loss for puzzles in the buffer (that are known to be good)
        if args.n_artificial > 0:
            sampled_fens, sampled_pvs, sampled_themes, sampled_ratings = buffer.sample(args.n_artificial)
            x_0_art_list = []
            for fen, pv in zip(sampled_fens, sampled_pvs):
                fen_tokens = tokenize_fen(fen)
                move_str = pv.split(" ")[0] if " " in pv else pv
                move_tokens = tokenize_move(move_str)
                x_0_art_list.append(fen_tokens + move_tokens)
            x_0_art = torch.tensor(x_0_art_list, dtype=torch.long, device=device)
            
            t = torch.rand(args.n_artificial, device=device)
            alpha_t = config.masking_schedule(t).unsqueeze(1)
            U = torch.rand((args.n_artificial, model.seq_length), device=device)
            x_t_art = torch.where(U < 1.0 - alpha_t, config.mask_token, x_0_art)
            
            if config.use_context:
                art_themes_one_hot = torch.from_numpy(theme_preprocessor.transform(sampled_themes)).to(device=device, dtype=torch.float32)
                art_scaled_ratings = scale_ratings(sampled_ratings).to(device=device, dtype=torch.float32)
            else:
                art_themes_one_hot, art_scaled_ratings = None, None

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits_art = model(
                    x_t_art,
                    art_themes_one_hot,
                    art_scaled_ratings,
                    checkpoint_activations=args.checkpoint_activations
                )
            
            loss_sup = args.sup_loss_coef * model.elbo_loss(t, logits_art, x_0_art, x_t_art).mean() / model.seq_length
            loss_sup.backward()
            
            total_loss += loss_sup.item()
            
            # Clean up SFT tensors
            del logits_art, loss_sup, x_0_art, x_t_art, t, U
            if config.use_context:
                del art_themes_one_hot, art_scaled_ratings
        
        total_norm += torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2).item()

        optimizer.step()
        scheduler.step()

    # Clean up dataloader/dataset before returning
    len_dataloader = len(dataloader)
    del dataset, dataloader
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    log_metrics(
        total_loss / args.ppo_epochs,
        total_norm / args.ppo_epochs,
        total_kl_divergence.mean().item(),
        n_clips / (len_dataloader * args.ppo_epochs),
        step,
        optimizer.param_groups[0]["lr"],
        reward=reward_val,
        entropy=total_entropy.mean().item(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--reference_path", type=str, required=True)
    parser.add_argument("--n_generations", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_artificial", type=int, default=4)
    parser.add_argument("--ppo_minibatch_size", type=int, default=4096)
    parser.add_argument("--checkpoint_activations", action="store_true", help="Use activation checkpointing to save GPU memory")
    # parser.add_argument("--save_period", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--kl_coef", type=float, default=0.03)
    parser.add_argument("--entropy_coef", type=float, default=0.03)
    parser.add_argument("--sup_loss_coef", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--ppo_epochs", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--lichess_distribution", type=bool, default=True)
    args = parser.parse_args()

    cpu_count = int(os.environ.get("SLURM_CPUS_PER_TASK")) * int(os.environ.get("SLURM_NTASKS")) - 2

    base_path = Path("./src")
    path = base_path / "runs" / "rl" / args.run_name
    path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(path)

    # engine = SimpleEngine.popen_uci(base_path / ".." / "Stockfish" / "src" / "stockfish")
    capacity = 200_000

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    checkpoint = torch.load(args.reference_path, map_location="cpu", weights_only=False)

    config = checkpoint["config"]
    config.schedule = "linear"
    config.masking_schedule = string_to_schedule(config.schedule)  # NOTE: remember that we are using a linear masking schedule

    buffer_folder = "rl_themes" if config.use_context else "rl"
    buffer = ReplayBuffer(capacity, base_path / "dataset" / buffer_folder)

    model = MaskedDiffusion(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device=device)

    reference_model = MaskedDiffusion(checkpoint["config"])
    reference_model.load_state_dict(checkpoint["model"])
    reference_model.to(device=device)
    reference_model.eval()
    reference_model.requires_grad_(False)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=3)

    for step in range(args.n_generations):
        train_ddpo(model, reference_model, optimizer, scheduler, config, device, args, step)
