from pathlib import Path
from copy import deepcopy
from datetime import datetime
import argparse
import os
from joblib import Parallel, delayed
import random

import torch
from torch.optim import AdamW, Muon
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group, all_reduce, all_gather_into_tensor, ReduceOp, gather, scatter
from chess.engine import SimpleEngine

from MaskedDiffusion.model import MaskedDiffusion
from rl.espo import espo_loss, generate_grouped_positions, generate_random_themes, compute_elbo
from tokenization.tokenization import theme_preprocessor, scale_ratings, tokens_to_fen
from metrics.themes import legal, get_unique_puzzle_from_fen, counter_intuitive
from metrics.cook import cook
from metrics.diversity_filtering import ReplayBuffer
from metrics.rewards import good_piece_counts, good_inter_batch_distances, good_intra_batch_distances


parser = argparse.ArgumentParser()
parser.add_argument("--distributed", action="store_true")
# parser.add_argument("--continue_from_checkpoint", action="store_true")

args = parser.parse_args()
distributed = args.distributed
# continue_from_checkpoint = args.continue_from_checkpoint

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

# ====================== LOGGING ======================
if master_process:
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    logging_path = base_path / "runs"/ "rl" / current_time
    writer = SummaryWriter(logging_path)

base_path = Path("./src")
checkpoint = torch.load(base_path / "supervised_checkpoints" / "model_0280000.pt", map_location="cpu", weights_only=False)

config = checkpoint["config"]
model = MaskedDiffusion(config)
model.load_state_dict(checkpoint["model"])
model.to(device=device)

reference_model = deepcopy(model)
reference_model.to(device=device)

if distributed:
    model = DistributedDataParallel(model, device_ids=[local_rank])

capacity = 200_000
buffer = ReplayBuffer(capacity, base_path / "dataset" / "rl")

n_gradient_updates_per_generation = 8  # https://arxiv.org/pdf/2512.03759 figure 5 (8 - 24 seems reasonable)
total_steps = 20  # 20_000
batch_size = 4  # 32  # batch_size * group_size == 64 works for 32g of vram
local_batch_size = batch_size // world_size
group_size = 8  # 8 - 64 probably good?
eps = 0.2  # from https://arxiv.org/pdf/1707.06347 page 6
beta = 0.03  # from https://arxiv.org/pdf/2510.23881 page 34

params_adam = [p for p in model.parameters() if p.ndim != 2]
params_muon = [p for p in model.parameters() if p.ndim == 2]
adam = AdamW(params_adam, lr=4e-3, weight_decay=0)  # 4e-3
muon = Muon(params_muon, lr=4e-3, weight_decay=0)  # 4e-3

engine = SimpleEngine.popen_uci(base_path / ".." / "Stockfish" / "src" / "stockfish")

def get_puzzle(fen):
    if fen is None: return None
    if not legal(fen): return None
    with SimpleEngine.popen_uci(base_path / ".." / "Stockfish" / "src" / "stockfish") as stockfish:
        puzzle = get_unique_puzzle_from_fen(fen, stockfish)
    return puzzle

cpu_count = int(os.environ.get("SLURM_CPUS_PER_TASK")) * int(os.environ.get("SLURM_NTASKS"))

def get_rewards(fen_tokens, themes, ratings):
    # TODO: add rating dependence
    legal_position = torch.zeros(len(fen_tokens), dtype=bool)
    unique_solution = torch.zeros(len(fen_tokens), dtype=bool)
    counter_intuitive_solution = torch.zeros(len(fen_tokens), dtype=bool)
    piece_counts = torch.zeros(len(fen_tokens), dtype=bool)
    inter_batch_fen_dist = torch.zeros(len(fen_tokens), dtype=bool)
    intra_batch_fen_dist = torch.zeros(len(fen_tokens), dtype=bool)
    inter_batch_pv_dist = torch.zeros(len(fen_tokens), dtype=bool)
    intra_batch_pv_dist = torch.zeros(len(fen_tokens), dtype=bool)
    themes_match = torch.zeros(len(fen_tokens), dtype=bool)

    fens = []
    for tokens in fen_tokens:
        try:
            fen = tokens_to_fen(tokens)
            fens.append(fen)
        except:
            fens.append(None)

    puzzles = list(Parallel(n_jobs=cpu_count)(delayed(get_puzzle)(fen) for fen in fens))  # do the puzzle computation in parallel as it is the most time consuming step

    group_size = len(fen_tokens) // len(ratings)
    for i, fen in enumerate(fens):
        theme, rating = themes[i // group_size], ratings[i // group_size]
        if fen is None:
            continue
        if not legal(fen):
            continue
        legal_position[i] = 1
        
        puzzle = puzzles[i]
        if puzzle is None:
            continue
        unique_solution[i] = 1
        counter_intuitive_solution[i] = counter_intuitive(fen, engine)
        piece_counts[i] = good_piece_counts(puzzle)

        sampled_fens, sampled_pvs = buffer.sample(32)
        pv = " ".join([move.uci() for move in puzzle.mainline])
        intra_batch_fen_dist[i], intra_batch_pv_dist[i] = good_intra_batch_distances(fen, pv, puzzles)
        inter_batch_fen_dist[i], inter_batch_pv_dist[i] = good_inter_batch_distances(fen, pv, sampled_fens, sampled_pvs)

        generation_themes = cook(puzzle, engine)
        themes_match[i] = len(set(generation_themes).intersection(theme)) > len(theme) / 2

    distance_rewards = inter_batch_fen_dist + inter_batch_pv_dist + intra_batch_fen_dist + intra_batch_pv_dist
    rewards = 2 * counter_intuitive_solution + 0.5 * piece_counts + 0.5 * themes_match + 0.5 * distance_rewards
    rewards = torch.where(unique_solution, rewards, 0)
    rewards = torch.where(legal_position, rewards, -2)

    log_data = {
        "legal_rate": legal_position.float().mean().item(),
        "uniqueness_rate": unique_solution.float().mean().item(),
        "counter_intuitive_rate": counter_intuitive_solution.float().mean().item(),
        "piece_counts": piece_counts.float().mean().item(),
        "themes_match_rate": themes_match.float().mean().item(),
        "dist_inter_fen": inter_batch_fen_dist.float().mean().item(),
        "dist_intra_fen": intra_batch_fen_dist.float().mean().item(),
        "dist_inter_pv": inter_batch_pv_dist.float().mean().item(),
        "dist_intra_pv": intra_batch_pv_dist.float().mean().item(),
    }
    for key, value in log_data.items():
        writer.add_scalar(f"Components/{key}", value, step)
    writer.add_scalar("Reward", rewards.mean().item(), step)

    return rewards

def save_state():
    checkpoint_path = base_path / "rl_checkpoints" / f"model_{step:07d}.pt"
    checkpoint = {
        "model": model.module.state_dict() if distributed else model.state_dict(),
        "config": config,
        "adam": adam.state_dict(),
        "muon": muon.state_dict(),
        "step": step
    }
    torch.save(checkpoint, checkpoint_path)

step = 0
end = False
while not end:
    themes, ratings = generate_random_themes(local_batch_size)
    ratings = ratings.to(device=device, dtype=torch.float32)
    themes_one_hot = torch.from_numpy(theme_preprocessor.transform(themes)).to(device=device, dtype=torch.float32)
    scaled_ratings = scale_ratings(ratings).to(device=device, dtype=torch.float32)

    # generate the fens from the old_model
    step_fens, step_themes, step_ratings = generate_grouped_positions(model, themes_one_hot, scaled_ratings, group_size, steps=128)

    # compute the rewards (on the master process)
    if distributed:
        gather_fens = [torch.zeros_like(step_fens) for _ in range(world_size)] if master_process else None
        gather_themes = [torch.zeros_like(themes_one_hot) for _ in range(world_size)] if master_process else None
        gather_ratings = [torch.zeros_like(ratings) for _ in range(world_size)] if master_process else None
        gather(step_fens, gather_list=gather_fens, dst=0)
        gather(themes_one_hot, gather_list=gather_themes, dst=0)
        gather(ratings, gather_list=gather_ratings, dst=0)
        if master_process:
            global_fens = torch.cat(gather_fens)
            global_themes = torch.cat(gather_themes)
            global_ratings = torch.cat(gather_ratings)
            global_themes_str = theme_preprocessor.inverse_transform(global_themes.cpu().numpy())
            global_rewards = get_rewards(global_fens.cpu(), global_themes_str, global_ratings.cpu()).to(device=device, dtype=torch.float32)
            reward_chunks = list(global_rewards.chunk(world_size, dim=0))
        rewards = torch.zeros(len(step_fens), device=device, dtype=torch.float32)
        scatter(rewards, scatter_list=reward_chunks if master_process else None, src=0)
    else:
        rewards = get_rewards(step_fens, themes, ratings)
    
    # precompute the elbos of the old model
    with torch.no_grad():
        reference_elbo, mask = compute_elbo(reference_model, step_fens, step_themes, step_ratings, return_mask=True)
        old_elbo = compute_elbo(model, step_fens, step_themes, step_ratings, mask=mask)

    # update the model many times for one generation (as generations and reward calculations are expensive)
    total_loss = 0
    total_norm = 0
    for substep in range(n_gradient_updates_per_generation):
        step += 1

        adam.zero_grad()
        muon.zero_grad()

        loss = espo_loss(model, reference_elbo, old_elbo, step_fens, step_themes, step_ratings, rewards, group_size, mask, eps=eps, beta=beta).mean()
        total_loss += loss

        loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        total_norm += norm

        adam.step()
        muon.step()

        if step >= total_steps:
            end = True
    
    if distributed:
        all_reduce(total_loss, op=ReduceOp.AVG)
        all_reduce(total_norm, op=ReduceOp.AVG)
    if master_process:
        writer.add_scalar("Loss", total_loss.item() / n_gradient_updates_per_generation, step)
        writer.add_scalar("Grad norm", total_norm.item() / n_gradient_updates_per_generation, step)

    if step // n_gradient_updates_per_generation % 1 == 0:
        save_state()

if master_process:
    writer.add_hparams({
        "batch_size": batch_size,
        "group_size": group_size,
        "n_gradient_updates_per_generation": n_gradient_updates_per_generation,
        "adam_lr": adam.param_groups[0]['lr'],
        "muon_lr": muon.param_groups[0]['lr'],
        "PPO_eps": eps,
        "PPO_beta": beta
    }, {
        "rewards": rewards.mean().item()
    })

engine.quit()
if master_process: writer.close()
if distributed: destroy_process_group()
