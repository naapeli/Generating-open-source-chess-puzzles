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
from concurrent.futures import ThreadPoolExecutor
import os
import random

from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp, gather, scatter

from MaskedDiffusion.model import MaskedDiffusion
from tokenization.tokenization import theme_preprocessor, scale_ratings, unscale_ratings, tokens_to_fen, tokens_to_move, tokenize_fen, tokenize_move
from metrics.themes import legal, get_unique_puzzle_from_fen, counter_intuitive
from metrics.diversity_filtering import ReplayBuffer
from metrics.rewards import good_piece_counts, inter_batch_distances, intra_batch_distances
from MaskingSchedule.MaskingSchedule import string_to_schedule

from rl.espo import compute_elbo, compute_elbo_basic, kl_estimate, entropy, generate_random_themes

def log_metrics(total_loss, grad_norm, kl_divergence, clips, step, lr, reward=None, entropy=None):
    if writer is None: return
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
    if writer is None: return
    try:
        board = chess.Board(fen)
        svg_data = svg.board(board, size=300)
        png_data = cairosvg.svg2png(bytestring=svg_data.encode("utf-8"))
        board_img = Image.open(io.BytesIO(png_data)).convert("RGB")
        text_height = 80
        info_pane = Image.new("RGB", (board_img.width, text_height), (255, 255, 255))
        draw = ImageDraw.Draw(info_pane)
        text_content = f"{rating}\n{themes}\n{fen}" if themes is not None else fen
        draw.text((10, 10), text_content, fill=(0, 0, 0))
        combined_img = np.vstack((np.array(board_img), np.array(info_pane)))
        
        writer.add_image(tag, combined_img, step, dataformats="HWC")
        writer.add_text(f"{tag}/fen", text_content, step)
    except Exception:
        pass

def get_reward(x_t, entropy_vals, config, step, themes_tokens=None, ratings=None):
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
    except Exception as e:
        print(e)
        return torch.zeros(batch_size, dtype=torch.float32, device=x_t.device)
        
    puzzles = [r[0] for r in results]
    best_moves = [r[1] for r in results]
    found_cp_losses = [r[2] for r in results]
    pvs = [r[3] for r in results]
    ci_solutions = [r[4] for r in results]
    ci_values = [r[5] for r in results]
    
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
        good_distances = (intra_batch_fen_dist[i] >= 6) and (intra_batch_pv_dist[i] >= 1) and (inter_batch_fen_dist[i] >= 6)
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

    pass_diversity_filtering = good_intra_fen & good_inter_fen & good_intra_pv & piece_counts  #  & (entropy_vals > 0.6)
    # pass_diversity_filtering = torch.ones_like(good_intra_fen)

    components = {
        "legal_rate": legal_position.float().mean().item(),
        "uniqueness_rate": unique_solution.float().mean().item(),
        "counter_intuitive_rate": counter_intuitive_solution.float().mean().item(),
        "counter_intuitive_values": counter_intuitive_values.mean().item(),
        "counter_intuitive_values_given_unique": counter_intuitive_values[is_valid].mean().item() if is_valid.any() else 0,
        "counter_intuitive_rate_given_unique": counter_intuitive_solution[is_valid].float().mean().item() if is_valid.any() else 0,
        "unique_and_counter_intuitive": unique_and_counter_intuitive.float().mean().item(),
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
    # rewards = torch.where(legal_position & pass_diversity_filtering & unique_and_counter_intuitive, 1.0, rewards)
    rewards = torch.where(legal_position & unique_solution, counter_intuitive_values, rewards)
    rewards = torch.where(legal_position & unique_and_counter_intuitive, 1.0, rewards)
    rewards = torch.where(~legal_position, -2.0, rewards)
    rewards = rewards.to(torch.float32)

    if writer is not None:
        for key, value in components.items():
            writer.add_scalar(f"Components/{key}", value, step)
    log_rewards = rewards.clone()
    index = torch.argmax(log_rewards).item()
    save_board(batch_fens[index], "Generations", step, themes[index], true_ratings[index])
    index = torch.argmin(log_rewards).item()
    save_board(batch_fens[index], "Worst_Generations", step, themes[index], true_ratings[index])
    index = torch.randint(0, batch_size, (1,)).item()
    save_board(batch_fens[index], "Random_Generations", step, themes[index], true_ratings[index])
    for index, reward in enumerate(log_rewards):
        if not (unique_solution[index] & counter_intuitive_solution[index] & pass_diversity_filtering[index]):
            continue
        save_board(batch_fens[index], "Puzzles", step, themes[index], true_ratings[index])
    
    return rewards.to(x_t.device, dtype=torch.float32)

def train_espo(model, reference_model, optimizer, scheduler, config, device, args, step, master_process, distributed, world_size, local_batch_size, local_n_arteficial):
    model.eval()
    
    if config.use_context:
        themes, ratings = generate_random_themes(local_batch_size, lichess_distribution=args.lichess_distribution)
        ratings = ratings.to(device=device, dtype=torch.float32)
        themes_one_hot = torch.from_numpy(theme_preprocessor.transform(themes)).to(device=device, dtype=torch.float32)
        scaled_ratings = scale_ratings(ratings).to(device=device, dtype=torch.float32)
        
        # Repeat interleave for group_size
        themes_one_hot = themes_one_hot.repeat_interleave(args.group_size, dim=0)
        scaled_ratings = scaled_ratings.repeat_interleave(args.group_size, dim=0)
        ratings = ratings.repeat_interleave(args.group_size, dim=0)
    else:
        themes, ratings = None, None
        themes_one_hot, scaled_ratings = None, None

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        sample_model = model.module if distributed else model
        step_fens, total_kl, total_entropy = sample_model.sample(themes_one_hot, scaled_ratings, steps=args.steps, batch_size=local_batch_size * args.group_size, temperature=args.temperature, generate_move_last=False, compute_kl=True, compute_entropy=True, ref_model=reference_model)
    
    local_exact_kl = total_kl.mean()
    local_exact_entropy = total_entropy.mean()
    if distributed:
        all_reduce(local_exact_kl, op=ReduceOp.AVG)
        all_reduce(local_exact_entropy, op=ReduceOp.AVG)
    exact_kl_val = local_exact_kl.item()
    exact_entropy_val = local_exact_entropy.item()

    if local_n_arteficial > 0:
        sampled_fens, sampled_pvs, sampled_themes, sampled_ratings = buffer.sample(local_n_arteficial)
        x_0_art_list = []
        for fen, pv in zip(sampled_fens, sampled_pvs):
            fen_tokens = tokenize_fen(fen)
            move_str = pv.split(" ")[0] if " " in pv else pv
            move_tokens = tokenize_move(move_str)
            x_0_art_list.append(fen_tokens + move_tokens)
        x_0_art = torch.tensor(x_0_art_list, dtype=torch.long, device=device)
        
        art_themes_one_hot = None
        art_scaled_ratings = None
        if config.use_context:
            art_themes_one_hot = torch.from_numpy(theme_preprocessor.transform(sampled_themes)).to(device=device, dtype=torch.float32)
            art_scaled_ratings = scale_ratings(sampled_ratings).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        # reference_elbo, mask, t = compute_elbo(reference_model, step_fens, themes=themes_one_hot, ratings=scaled_ratings, return_mask=True)
        # old_elbo = compute_elbo(model, step_fens, themes=themes_one_hot, ratings=scaled_ratings, mask=mask)
        old_elbo, mask, t = compute_elbo(model, step_fens, themes=themes_one_hot, ratings=scaled_ratings, return_mask=True)

    # entropy_vals = entropy(old_elbo, step_fens.shape[1])
    
    if distributed:
        gather_fens = [torch.zeros_like(step_fens) for _ in range(world_size)] if master_process else None
        gather_entropies = [torch.zeros_like(total_entropy) for _ in range(world_size)] if master_process else None
        
        gather(step_fens, gather_list=gather_fens, dst=0)
        gather(total_entropy, gather_list=gather_entropies, dst=0)

        if config.use_context:
            gather_themes = [torch.zeros_like(themes_one_hot) for _ in range(world_size)] if master_process else None
            gather_ratings = [torch.zeros_like(ratings) for _ in range(world_size)] if master_process else None
            gather(themes_one_hot, gather_list=gather_themes, dst=0)
            gather(ratings, gather_list=gather_ratings, dst=0)

        if master_process:
            global_fens = torch.cat(gather_fens)
            global_entropies = torch.cat(gather_entropies)
            global_themes = torch.cat(gather_themes) if config.use_context else None
            global_ratings = torch.cat(gather_ratings) if config.use_context else None
            global_rewards = get_reward(global_fens, global_entropies.cpu(), config, step, themes_tokens=global_themes, ratings=global_ratings)
            reward_chunks = list(global_rewards.chunk(world_size, dim=0))
        rewards_gen = torch.zeros(len(step_fens), device=device, dtype=torch.float32)
        scatter(rewards_gen, scatter_list=reward_chunks if master_process else None, src=0)
    else:
        rewards_gen = get_reward(step_fens, total_entropy.cpu(), config, step, themes_tokens=themes_one_hot, ratings=ratings)
    
    rewards_gen = rewards_gen - args.beta * total_kl + args.entropy_coef * total_entropy

    model.train()
    total_loss = 0
    total_norm = 0
    total_clips = 0

    for substep in range(args.n_gradient_updates_per_generation):
        optimizer.zero_grad()
        
        elbo_gen = compute_elbo(model, step_fens, themes=themes_one_hot, ratings=scaled_ratings, mask=mask, return_mask=False)
        if local_n_arteficial > 0:
            elbo_art = compute_elbo_basic(model, x_0_art, themes=art_themes_one_hot, ratings=art_scaled_ratings, n_mc=1)
            loss_sft = -elbo_art.mean() / step_fens.shape[1]
        else:
            loss_sft = torch.tensor(0.0, device=device)
            
        sequence_length = step_fens.shape[1]

        # kl = kl_estimate(elbo_gen, reference_elbo).mean() / sequence_length

        r_gen = rewards_gen.detach()
        
        r_gen_grouped = r_gen.reshape(local_batch_size, args.group_size)
        if args.group_size > 1:
            mean_other_rewards = (r_gen_grouped.sum(dim=1, keepdim=True) - r_gen_grouped) / (args.group_size - 1)
        else:
            mean_other_rewards = torch.zeros_like(r_gen_grouped)
        advantages_grouped = r_gen_grouped - mean_other_rewards
        advantages = advantages_grouped.reshape(-1).to(device)
        
        rho = torch.exp((elbo_gen - old_elbo) / sequence_length)
        coef_1 = rho * advantages
        coef_2 = torch.clamp(rho, 1 - args.eps, 1 + args.eps) * advantages
        loss_ppo = -torch.minimum(coef_1, coef_2).mean()
        
        is_clipped = (coef_2 < coef_1).flatten()
        
        loss = loss_ppo + args.gamma * loss_sft  #  + args.beta * kl
        
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        total_norm += norm.item()
        total_clips += is_clipped.float().mean().item()

    if distributed:
        t_total_loss = torch.tensor(total_loss, device=device)
        t_total_norm = torch.tensor(total_norm, device=device)
        t_total_clips = torch.tensor(total_clips, device=device)
        t_rewards_mean = torch.tensor(rewards_gen.float().mean().item(), device=device)

        all_reduce(t_total_loss, op=ReduceOp.AVG)
        all_reduce(t_total_norm, op=ReduceOp.AVG)
        all_reduce(t_total_clips, op=ReduceOp.AVG)
        all_reduce(t_rewards_mean, op=ReduceOp.AVG)

        total_loss = t_total_loss.item()
        total_norm = t_total_norm.item()
        total_clips = t_total_clips.item()
        rewards_mean = t_rewards_mean.item()
    else:
        rewards_mean = rewards_gen.float().mean().item()

    if master_process:
        log_metrics(
            total_loss / args.n_gradient_updates_per_generation,
            total_norm / args.n_gradient_updates_per_generation,
            exact_kl_val,
            total_clips / args.n_gradient_updates_per_generation,
            step,
            optimizer.param_groups[0]["lr"],
            reward=rewards_mean,
            entropy=exact_entropy_val
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--reference_path", type=str, required=True)
    parser.add_argument("--checkpoint_model", type=str, default=None)
    parser.add_argument("--save_period", type=int, default=2000)
    parser.add_argument("--n_generations", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_arteficial", type=int, default=4)
    parser.add_argument("--n_gradient_updates_per_generation", type=int, default=1)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument("--entropy_coef", type=float, default=0.03)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--lichess_distribution", type=bool, default=True)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--group_size", type=int, default=4)
    args = parser.parse_args()
    
    distributed = int(os.environ.get("SLURM_GPUS")) > 1
    assert int(os.environ.get("SLURM_GPUS")) == int(os.environ.get("SLURM_NTASKS")), "there must be as many tasks as gpus"
    
    if distributed:
        assert torch.cuda.is_available()
        local_rank = int(os.environ["SLURM_PROCID"])
        rank = local_rank
        world_size = int(os.environ["SLURM_NTASKS"])

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        master_process = rank == 0
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        master_process = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(rank)
    torch.manual_seed(rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rank)

    local_batch_size = args.batch_size // world_size
    local_n_arteficial = args.n_arteficial // world_size

    cpu_count = int(os.environ.get("SLURM_CPUS_PER_TASK", "1")) * int(os.environ.get("SLURM_NTASKS", "1"))

    base_path = Path("./src")
    path = base_path / "runs" / "rl" / "final_large_runs" / args.run_name
    if master_process:
        path.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(path)
    else:
        writer = None

    continue_from_checkpoint = args.checkpoint_model is not None
    reference_checkpoint = torch.load(args.reference_path, map_location="cpu", weights_only=False)
    if continue_from_checkpoint:
        checkpoint = torch.load(path / args.checkpoint_model, map_location="cpu", weights_only=False)
    else:
        checkpoint = reference_checkpoint

    config = checkpoint["config"]

    capacity = 200_000
    buffer_folder = "rl_themes" if config.use_context else "rl"
    buffer = ReplayBuffer(capacity, base_path / "dataset" / buffer_folder)

    config.schedule = "linear"
    config.masking_schedule = string_to_schedule(config.schedule)
    
    model = MaskedDiffusion(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device=device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    reference_config = reference_checkpoint["config"]
    reference_config.schedule = "linear"  # make the masking schedule of the reference model match the training masking schedule
    reference_config.masking_schedule = string_to_schedule(reference_config.schedule)
    reference_model = MaskedDiffusion(reference_config)
    reference_model.load_state_dict(reference_checkpoint["model"])
    reference_model.to(device=device)
    reference_model.eval()
    reference_model.requires_grad_(False)

    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01)
    if continue_from_checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    start_step = checkpoint.get("step", 0) if continue_from_checkpoint else 0
    scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=3, last_epoch=start_step if continue_from_checkpoint else -1)

    def save_state(step_val):
        checkpoint_path = path / f"model_{step_val:07d}.pt"
        save_dict = {
            "model": model.module.state_dict() if distributed else model.state_dict(),
            "config": config,
            "optimizer": optimizer.state_dict(),
            "step": step_val
        }
        torch.save(save_dict, checkpoint_path)

    for step in range(start_step, args.n_generations):
        train_espo(model, reference_model, optimizer, scheduler, config, device, args, step, master_process, distributed, world_size, local_batch_size, local_n_arteficial)
        
        if master_process and step > 0 and step % args.save_period == 0:
            save_state(step)

    if master_process:
        save_state(args.n_generations)
        writer.close()
    if distributed: destroy_process_group()
