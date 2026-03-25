from pathlib import Path
from copy import deepcopy
from datetime import datetime
import argparse
import os
from joblib import Parallel, delayed
import random
import io

import torch
from torch.optim import AdamW, Muon
from torch.optim.lr_scheduler import LinearLR
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp, gather, scatter
import chess
from chess import svg
from chess.engine import SimpleEngine
import cairosvg
from PIL import Image, ImageDraw
import numpy as np

from MaskedDiffusion.model import MaskedDiffusion
from RatingModel.model import RatingModel
from rl.espo import espo_loss, generate_grouped_positions, generate_random_themes, compute_elbo, compute_elbo_basic, theme_reward, critic_free_ppo_loss, entropy
from tokenization.tokenization import theme_preprocessor, scale_ratings, unscale_ratings, tokens_to_fen, tokenize_fen
from metrics.themes import legal, get_unique_puzzle_from_fen, counter_intuitive
from metrics.cook import cook
from metrics.diversity_filtering import ReplayBuffer
from metrics.rewards import good_piece_counts, good_inter_batch_distances, good_intra_batch_distances


parser = argparse.ArgumentParser()
parser.add_argument("--distributed", action="store_true")
parser.add_argument("--continue_from_checkpoint", action="store_true")

args = parser.parse_args()
distributed = args.distributed
continue_from_checkpoint = args.continue_from_checkpoint

base_path = Path("./src")

# ====================== DEVICE ======================
if distributed:
    assert torch.cuda.is_available()
    local_rank = int(os.environ["SLURM_PROCID"])
    rank = local_rank  # should be something different with multiple nodes
    world_size = int(os.environ["SLURM_NTASKS"])

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)
    master_process = rank == 0
else:
    rank = 0
    local_rank = 0
    world_size = 1
    master_process = True
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if device.startswith("cuda") else "cpu"
device = torch.device(device)

# ====================== SEED AND PRECISION ======================
random.seed(rank)
torch.manual_seed(rank)
if torch.cuda.is_available():
    torch.cuda.manual_seed(rank)
torch.set_float32_matmul_precision("high")

# ====================== CONFIG ======================
n_gradient_updates_per_generation = 1  # https://arxiv.org/pdf/2512.03759 figure 5 (8 - 24 seems reasonable) https://arxiv.org/pdf/2510.23881 uses 1
total_steps = 2_000 * n_gradient_updates_per_generation  # 20_000
batch_size = 16
local_batch_size = batch_size // world_size
group_size = 4
eps = 0.3  # from https://arxiv.org/pdf/1707.06347 page 6
beta = 3e-6  # 0  # 3e-2  # 1e-3  # 5e-4  # 1e-4
n_positions_added = 16  # matters only when group_size == 1
local_n_positions_added = n_positions_added // world_size
lichess_distribution = True
save_period = 500 * n_gradient_updates_per_generation

save_checkpoints = False
save_path = base_path / "rl_checkpoints" / "lichessDist"
reference_checkpoint_path = base_path / "supervised_checkpoints" / "model_0940000.pt"


# ====================== LOGGING ======================
if master_process:
    if continue_from_checkpoint:
        logging_path = base_path / "runs"/ "rl" / "espo" / "GoodResultsCounterIntuitivenessDoesNotImprove"
    else:
        # current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        # logging_path = base_path / "runs"/ "rl" / "espo" / current_time
        logging_path = base_path / "runs"/ "rl" / "espo" / "betaSearchV6" / f"{beta:.1e}"
    writer = SummaryWriter(logging_path)

reference_checkpoint = torch.load(reference_checkpoint_path, map_location="cpu", weights_only=False)
if continue_from_checkpoint:
    checkpoint = torch.load(base_path / "rl_checkpoints" / "model_0039360.pt", map_location="cpu", weights_only=False)
else:
    checkpoint = reference_checkpoint

config = checkpoint["config"]
model = MaskedDiffusion(config)
model.load_state_dict(checkpoint["model"])
model.to(device=device)

config.lr = 3e-4  # 3e-5  # 1e-4
config.weight_decay = 1e-5  # 1e-5

if master_process:
    rating_model_checkpoint = torch.load(base_path / "rating_model_checkpoints" / "model_0063000.pt", map_location="cpu", weights_only=False)
    rating_model = RatingModel(rating_model_checkpoint["config"])
    rating_model.load_state_dict(rating_model_checkpoint["model"])
    rating_model.to(device=device)

reference_model = MaskedDiffusion(reference_checkpoint["config"])
reference_model.load_state_dict(reference_checkpoint["model"])
reference_model.to(device=device)

if distributed:
    model = DistributedDataParallel(model, device_ids=[local_rank])

capacity = 200_000
buffer = ReplayBuffer(capacity, base_path / "dataset" / "rl")

params_adam = [p for p in model.parameters() if p.ndim != 2]
params_muon = [p for p in model.parameters() if p.ndim == 2]
adam = AdamW(params_adam, lr=config.lr, weight_decay=config.weight_decay)
muon = Muon(params_muon, lr=config.lr, weight_decay=config.weight_decay)

if continue_from_checkpoint:
    adam.load_state_dict(checkpoint["adam"])
    muon.load_state_dict(checkpoint["muon"])

adam_scheduler = LinearLR(adam, start_factor=0.01, end_factor=1, total_iters=20, last_epoch=checkpoint["step"] if continue_from_checkpoint else -1)
muon_scheduler = LinearLR(muon, start_factor=0.01, end_factor=1, total_iters=20, last_epoch=checkpoint["step"] if continue_from_checkpoint else -1)

cpu_count = int(os.environ.get("SLURM_CPUS_PER_TASK")) * int(os.environ.get("SLURM_NTASKS"))

engine = SimpleEngine.popen_uci(base_path / ".." / "Stockfish" / "src" / "stockfish")

# ====================== REWARDS ======================

def get_puzzle(fen):
    if fen is None: return None
    if not legal(fen): return None
    with SimpleEngine.popen_uci(base_path / ".." / "Stockfish" / "src" / "stockfish") as stockfish:
        puzzle = get_unique_puzzle_from_fen(fen, stockfish)
    return puzzle

def save_board(fen, themes, rating, tag):
    try:
        board = chess.Board(fen)
        svg_data = svg.board(board, size=300)
        png_data = cairosvg.svg2png(bytestring=svg_data.encode("utf-8"))
        board_img = Image.open(io.BytesIO(png_data)).convert("RGB")
        text_height = 80
        info_pane = Image.new("RGB", (board_img.width, text_height), (255, 255, 255))
        draw = ImageDraw.Draw(info_pane)
        text_content = f"{rating}\n{themes}\n{fen}"
        draw.text((10, 10), text_content, fill=(0, 0, 0))
        combined_img = np.vstack((np.array(board_img), np.array(info_pane)))
        
        writer.add_image(tag, combined_img, step, dataformats="HWC")
        writer.add_text(f"{tag}/fen", text_content, step)
    except Exception:
        pass

def get_rewards(fen_tokens, theme_tokens, ratings, is_generated, elbo):
    legal_position = torch.zeros(len(fen_tokens), dtype=bool)
    unique_solution = torch.zeros(len(fen_tokens), dtype=bool)
    counter_intuitive_solution = torch.zeros(len(fen_tokens), dtype=bool)
    piece_counts = torch.zeros(len(fen_tokens), dtype=bool)
    inter_batch_fen_dist = torch.zeros(len(fen_tokens), dtype=bool)
    intra_batch_fen_dist = torch.zeros(len(fen_tokens), dtype=bool)
    inter_batch_pv_dist = torch.zeros(len(fen_tokens), dtype=bool)
    intra_batch_pv_dist = torch.zeros(len(fen_tokens), dtype=bool)
    themes_match = torch.zeros(len(fen_tokens), dtype=bool)
    rating_penalty = torch.zeros(len(fen_tokens), dtype=torch.float32)

    _, sequence_length = fen_tokens.shape
    entropies = entropy(elbo, sequence_length)
    entropy_reward = entropies > 0.6

    is_generated = is_generated.cpu()

    group_size = len(fen_tokens) // len(ratings)
    themes = theme_preprocessor.inverse_transform(theme_tokens.cpu().numpy())
    true_ratings = ratings.repeat_interleave(group_size, dim=0)

    device = fen_tokens.device
    fen_tokens_cpu = fen_tokens.cpu()

    fens = []
    for tokens in fen_tokens_cpu:
        try:
            fen = tokens_to_fen(tokens)
            fens.append(fen)
        except:
            fens.append(None)

    try:
        puzzles = list(Parallel(n_jobs=cpu_count)(delayed(get_puzzle)(fen) for fen in fens))
    except TimeoutError:
        print("Stockfish timed out")
        return torch.zeros(len(fen_tokens), dtype=torch.float32)

    valid_indices = []
    valid_gen_themes = []

    for i, fen in enumerate(fens):
        theme = themes[i // group_size]
        if fen is None or not legal(fen):
            continue
        legal_position[i] = 1

        piece_counts[i] = good_piece_counts(fen)
        
        puzzle = puzzles[i]
        if puzzle is None:
            continue
        unique_solution[i] = 1
        counter_intuitive_solution[i] = counter_intuitive(fen, engine)

        sampled_fens, sampled_pvs, _, _ = buffer.sample(2000)
        pv = " ".join([move.uci() for move in puzzle.mainline])
        intra_batch_fen_dist[i], intra_batch_pv_dist[i] = good_intra_batch_distances(fen, pv, puzzles, i)
        inter_batch_fen_dist[i], inter_batch_pv_dist[i] = good_inter_batch_distances(fen, pv, sampled_fens, sampled_pvs)

        generation_themes = cook(puzzle, engine)
        themes_match[i] = theme_reward(theme, generation_themes)

        # if the position returns a high reward, add it to the buffer
        # good_distances = intra_batch_fen_dist[i] and intra_batch_pv_dist[i] and inter_batch_fen_dist[i] and inter_batch_pv_dist[i]
        # if counter_intuitive_solution[i] and piece_counts[i] and good_distances and themes_match[i]:
        #     buffer.add(fen, pv, generation_themes, true_ratings[i].cpu().item())

        valid_indices.append(i)
        valid_gen_themes.append(generation_themes)

    if valid_indices:
        gen_theme_tokens = torch.tensor(theme_preprocessor.transform(valid_gen_themes), dtype=theme_tokens.dtype, device=device)
        valid_fen_tokens = fen_tokens[valid_indices]
        
        with torch.no_grad():
            predicted_ratings = unscale_ratings(rating_model(valid_fen_tokens, gen_theme_tokens))
            valid_true_ratings = true_ratings[valid_indices].to(device)
            
            penalties = -torch.clamp(torch.abs(predicted_ratings - valid_true_ratings) / 1000, 0, 1)
            rating_penalty[valid_indices] = penalties.cpu()

    intra_distances = intra_batch_fen_dist * intra_batch_pv_dist
    inter_distances = inter_batch_fen_dist * inter_batch_pv_dist
    all_distances = intra_distances * inter_distances
    
    # rewards = torch.ones(len(fen_tokens), dtype=torch.float32)# + 0.1 * rating_penalty
    # # rewards = torch.where(themes_match & piece_counts & all_distances & unique_solution & counter_intuitive_solution, rewards, 0)
    # rewards = torch.where(unique_solution & counter_intuitive_solution, rewards, 0)
    # rewards = torch.where(legal_position, rewards, -2)
    pass_diversity_filtering = entropy_reward & intra_distances & inter_batch_fen_dist & piece_counts
    rewards = torch.zeros(len(fen_tokens), dtype=torch.float32)
    rewards = torch.where(unique_solution & pass_diversity_filtering, 0.01, rewards)  # 1.0  # 0.0  # 0.5
    rewards = torch.where(unique_solution & counter_intuitive_solution & pass_diversity_filtering, 1.0, rewards)  # 3.0  # 1.0
    rewards = torch.where(legal_position, rewards, -2.0)  # -1.0

    # save the images of some positions
    log_rewards = rewards.clone()
    log_rewards[~is_generated] = -float("inf")
    index = torch.argmax(log_rewards).item()
    save_board(fens[index], themes[index // group_size], ratings[index // group_size], "Generations")
    for index, reward in enumerate(log_rewards):
        if not (unique_solution[i] & counter_intuitive_solution[i] & themes_match[i]):
            continue
        save_board(fens[index], themes[index // group_size], ratings[index // group_size], "Puzzles")
    gen_indices = torch.nonzero(is_generated).squeeze(1)
    index = gen_indices[torch.median(rewards[gen_indices], dim=0).indices].item()
    save_board(fens[index], themes[index // group_size], ratings[index // group_size], "Median")

    is_valid = torch.zeros(len(fen_tokens), dtype=torch.bool)
    is_valid[valid_indices] = True
    valid_mask = is_valid & is_generated

    log_data = {
        "legal_rate": legal_position[is_generated].float().mean().item(),
        "uniqueness_rate": unique_solution[is_generated].float().mean().item(),
        "counter_intuitive_rate": counter_intuitive_solution[is_generated].float().mean().item(),
        "counter_intuitive_rate_given_unique": counter_intuitive_solution[valid_mask].float().mean().item() if valid_mask.any() else 0,
        "entropy": entropies[is_generated].float().mean().item(),
        "piece_counts": piece_counts[is_generated & legal_position].float().mean().item(),
        "themes_match_rate": themes_match[valid_mask].float().mean().item() if valid_mask.any() else 0,
        "dist_inter_fen": inter_batch_fen_dist[valid_mask].float().mean().item() if valid_mask.any() else 0,
        "dist_intra_fen": intra_batch_fen_dist[valid_mask].float().mean().item() if valid_mask.any() else 0,
        "dist_inter_pv": inter_batch_pv_dist[valid_mask].float().mean().item() if valid_mask.any() else 0,
        "dist_intra_pv": intra_batch_pv_dist[valid_mask].float().mean().item() if valid_mask.any() else 0,
        "intra_dist": intra_distances[valid_mask].float().mean().item() if valid_mask.any() else 0,
        "inter_dist": inter_distances[valid_mask].float().mean().item() if valid_mask.any() else 0,
        "all_dist": all_distances[valid_mask].float().mean().item() if valid_mask.any() else 0,
        "rating_abs_diff": (-1000 * rating_penalty[valid_mask]).float().mean().item() if valid_mask.any() else 1000,
        "pass_diversity_filtering": pass_diversity_filtering[is_generated].float().mean().item()
    }
    
    for key, value in log_data.items():
        writer.add_scalar(f"Components/{key}", value, step)
    writer.add_scalar("Reward", rewards[is_generated].float().mean().item(), step)

    return rewards

def save_state():
    checkpoint_path = save_path / f"model_{step:07d}.pt"
    checkpoint = {
        "model": model.module.state_dict() if distributed else model.state_dict(),
        "config": config,
        "adam": adam.state_dict(),
        "muon": muon.state_dict(),
        "step": step
    }
    torch.save(checkpoint, checkpoint_path)

# ====================== TRAINING LOOP ======================

step = checkpoint["step"] if continue_from_checkpoint else 0
end = False
while not end:
    themes, ratings = generate_random_themes(local_batch_size, lichess_distribution=lichess_distribution)
    ratings = ratings.to(device=device, dtype=torch.float32)
    themes_one_hot = torch.from_numpy(theme_preprocessor.transform(themes)).to(device=device, dtype=torch.float32)
    scaled_ratings = scale_ratings(ratings).to(device=device, dtype=torch.float32)

    # generate the fens from the old_model
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):  # make the sampling a little faster with less precision
        step_fens, step_themes, step_ratings = generate_grouped_positions(model, themes_one_hot, scaled_ratings, group_size, steps=128)
    
    is_generated = torch.ones(len(step_fens), dtype=torch.bool, device=device)
    
    # if we use critic-free PPO, add positions from the replay buffer into the generated positions
    if group_size == 1:
        sampled_fens, _, sampled_themes, sampled_ratings = buffer.sample(local_n_positions_added)

        sampled_fens = torch.tensor([tokenize_fen(sampled_fen) for sampled_fen in sampled_fens]).to(device=device)
        sampled_themes = torch.from_numpy(theme_preprocessor.transform(sampled_themes)).to(device=device, dtype=torch.float32)
        themes_one_hot = torch.cat([themes_one_hot, sampled_themes], dim=0)
        ratings = torch.cat([ratings, sampled_ratings.to(device=device)], dim=0)
        sampled_ratings = scale_ratings(sampled_ratings).to(device=device, dtype=torch.float32)
        
        is_generated = torch.cat([is_generated, torch.zeros(local_n_positions_added, dtype=torch.bool, device=device)])
        step_fens = torch.cat([step_fens, sampled_fens], dim=0)
        step_themes = torch.cat([step_themes, sampled_themes], dim=0)
        step_ratings = torch.cat([step_ratings, sampled_ratings], dim=0)

    # precompute the elbos of the old model
    with torch.no_grad():
        reference_elbo, mask, t = compute_elbo(reference_model, step_fens, step_themes, step_ratings, return_mask=True)
        old_elbo = compute_elbo(model, step_fens, step_themes, step_ratings, mask=mask)
        # reference_elbo, mask, t = compute_elbo_basic(reference_model, step_fens, step_themes, step_ratings, return_mask=True)
        # old_elbo = compute_elbo_basic(model, step_fens, step_themes, step_ratings, mask=mask, t=t)

    # compute the rewards (on the master process)
    if distributed:
        gather_is_generated = [torch.zeros_like(is_generated) for _ in range(world_size)] if master_process else None
        gather_fens = [torch.zeros_like(step_fens) for _ in range(world_size)] if master_process else None
        gather_themes = [torch.zeros_like(themes_one_hot) for _ in range(world_size)] if master_process else None
        gather_ratings = [torch.zeros_like(ratings) for _ in range(world_size)] if master_process else None
        gather_elbos = [torch.zeros_like(old_elbo) for _ in range(world_size)] if master_process else None
        gather(is_generated, gather_list=gather_is_generated, dst=0)
        gather(step_fens, gather_list=gather_fens, dst=0)
        gather(themes_one_hot, gather_list=gather_themes, dst=0)
        gather(ratings, gather_list=gather_ratings, dst=0)
        gather(old_elbo, gather_list=gather_elbos, dst=0)
        if master_process:
            global_is_generated = torch.cat(gather_is_generated)
            global_fens = torch.cat(gather_fens)
            global_themes = torch.cat(gather_themes)
            global_ratings = torch.cat(gather_ratings)
            global_elbos = torch.cat(gather_elbos)
            global_rewards = get_rewards(global_fens, global_themes, global_ratings, global_is_generated, global_elbos).to(device=device, dtype=torch.float32)
            reward_chunks = list(global_rewards.chunk(world_size, dim=0))
        rewards = torch.zeros(len(step_fens), device=device, dtype=torch.float32)
        scatter(rewards, scatter_list=reward_chunks if master_process else None, src=0)
    else:
        rewards = get_rewards(step_fens, themes_one_hot, ratings, is_generated, old_elbo)

    # update the model many times for one generation (as generations and reward calculations are expensive)
    total_loss = 0
    total_norm = 0
    total_kl = 0
    total_clips = 0
    for substep in range(n_gradient_updates_per_generation):
        step += 1

        adam.zero_grad()
        muon.zero_grad()

        if group_size == 1:
            loss, kl, is_clipped = critic_free_ppo_loss(model, reference_elbo, old_elbo, step_fens, step_themes, step_ratings, rewards, group_size, mask=mask, t=t, eps=eps, beta=beta)
        else:
            loss, kl, is_clipped = espo_loss(model, reference_elbo, old_elbo, step_fens, step_themes, step_ratings, rewards, group_size, mask=mask, t=t, eps=eps, beta=beta)
        loss = loss.mean()
        total_loss += loss
        total_kl += kl.mean()
        total_clips += is_clipped.float().mean()

        loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        total_norm += norm

        adam.step()
        muon.step()
        adam_scheduler.step()
        muon_scheduler.step()

    if step >= total_steps:
        end = True
    
    if distributed:
        all_reduce(total_loss, op=ReduceOp.AVG)
        all_reduce(total_norm, op=ReduceOp.AVG)
        all_reduce(total_kl, op=ReduceOp.AVG)
        all_reduce(total_clips, op=ReduceOp.AVG)
    if master_process:
        writer.add_scalar("Loss/Loss", total_loss.item() / n_gradient_updates_per_generation, step)
        writer.add_scalar("Loss/Grad norm", total_norm.item() / n_gradient_updates_per_generation, step)
        writer.add_scalar("Loss/KL divergence", total_kl.item() / n_gradient_updates_per_generation, step)
        writer.add_scalar("Loss/learning_rate", adam.param_groups[0]["lr"], step)
        writer.add_scalar("Loss/Clips", total_clips.item() / n_gradient_updates_per_generation, step)

    if step % save_period == 0 and save_checkpoints:
        save_state()

# if master_process:
#     writer.add_hparams({
#         "batch_size": batch_size,
#         "group_size": group_size,
#         "n_gradient_updates_per_generation": n_gradient_updates_per_generation,
#         "lr": config.lr,
#         "weight_decay": config.weight_decay,
#         "PPO_eps": eps,
#         "PPO_beta": beta
#     }, {
#         "rewards": rewards.mean().item()
#     })

engine.quit()
if master_process: writer.close()
if distributed: destroy_process_group()
