import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp
import chess
from chess import svg
from chess.engine import SimpleEngine, Limit
import cairosvg
from PIL import Image, ImageDraw
import numpy as np
from joblib import Parallel, delayed

from pathlib import Path
from datetime import datetime
import os
import argparse
import random
import io

from MaskedDiffusion.model import MaskedDiffusion
from Config import Config
from tokenization.tokenization import tokens_to_fen, tokens_to_move, scale_ratings, unscale_ratings, theme_preprocessor
from metrics.themes import legal, get_unique_puzzle_from_fen, counter_intuitive
from metrics.cook import cook
from metrics.diversity_filtering import ReplayBuffer
from metrics.rewards import good_piece_counts, good_inter_batch_distances, good_intra_batch_distances
from RatingModel.model import RatingModel
from rl.espo import compute_elbo_basic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--checkpoint_name", type=str, default=None)
    parser.add_argument("--starting_model_path", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--validation_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100_000)
    args = parser.parse_args()
    
    distributed = args.distributed
    continue_from_checkpoint = args.checkpoint_name is not None

    base_path = Path("./src")
    
    run_name = args.run_name
    if not run_name:
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    logging_path = base_path / "runs" / "best_move" / run_name

    # ====================== DEVICE ======================
    if distributed:
        assert torch.cuda.is_available()
        local_rank = int(os.environ["SLURM_PROCID"])
        rank = local_rank
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

    # ====================== CONFIG ======================
    if continue_from_checkpoint:
        checkpoint = torch.load(logging_path / args.checkpoint_name, map_location="cpu", weights_only=False)
    else:
        checkpoint = torch.load(base_path / args.starting_model_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config.use_context == True:
        from warnings import warn
        warn("use_context is True, but we are training without conditioning. Setting use_context to False")
        config.use_context = False
    
    # ====================== SEED AND PRECISION ======================
    torch.manual_seed(rank)
    random.seed(rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rank)
    torch.set_float32_matmul_precision("high")

    # ====================== LOGGING ======================
    if master_process:
        logging_path.mkdir(parents=True, exist_ok=True)
        train_writer = SummaryWriter(logging_path / "train")
        validation_writer = SummaryWriter(logging_path / "validation")

    # ====================== DATASET ======================
    dataset_path = base_path / "dataset" / "normal_positions" / "large_position_dataset.pt"
    full_dataset = torch.load(dataset_path, weights_only=False, map_location="cpu")
    
    total_size = len(full_dataset)
    n_validation = 50_000
    all_indices = list(range(total_size))
    rng = random.Random(42)
    rng.shuffle(all_indices)
    val_indices = all_indices[:n_validation]
    train_indices = all_indices[n_validation:]
    trainset = Subset(full_dataset, train_indices)
    testset = Subset(full_dataset, val_indices)
    
    trainsampler = DistributedSampler(trainset, shuffle=True, rank=rank, num_replicas=world_size) if distributed else None
    validationsampler = DistributedSampler(testset, shuffle=False, rank=rank, num_replicas=world_size) if distributed else None
    
    local_batch_size = config.batch_size // world_size
    trainloader = DataLoader(trainset, batch_size=local_batch_size, shuffle=not distributed, sampler=trainsampler, num_workers=4 if distributed else 0, pin_memory=distributed)
    validationloader = DataLoader(testset, batch_size=local_batch_size, shuffle=False, sampler=validationsampler, num_workers=4 if distributed else 0, pin_memory=distributed)

    # ====================== MODEL ======================
    model = MaskedDiffusion(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device=device)

    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    model = torch.compile(model)

    # ====================== OPTIMIZER ======================
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler1 = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=1000)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=config.n_steps, eta_min=0.1 * config.lr)
    lr_scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[1000], last_epoch=checkpoint["step"] if continue_from_checkpoint else -1)
    if continue_from_checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception:
            pass

    # ====================== ENGINE & RATING MODEL (for metrics) ======================
    if master_process:
        stockfish_path = base_path / ".." / "Stockfish" / "src" / "stockfish"
        engine = SimpleEngine.popen_uci(stockfish_path)
        
        rating_model_checkpoint = torch.load(base_path / "runs" / "rating_model" / "v1" / "model_0063000.pt", map_location="cpu", weights_only=False)
        rating_model = RatingModel(rating_model_checkpoint["config"])
        rating_model.load_state_dict(rating_model_checkpoint["model"])
        rating_model.to(device=device)
        rating_model.eval()

        buffer = ReplayBuffer(200_000, base_path / "dataset" / "rl")
        cpu_count = int(os.environ.get("SLURM_CPUS_PER_TASK", 1)) * world_size
    else:
        engine = None
        rating_model = None
        buffer = None
        cpu_count = 1

    # ====================== LOSS FUNCTION ======================
    def compute_loss(model, fens, moves):
        tokens = torch.cat([fens, moves], dim=1)
        batch_size = len(fens)
        t = ((torch.rand(1) + torch.arange(1, batch_size + 1, 1) / batch_size) % 1).to(device)
        t = t ** (1/7)  # Skew t towards 1.0 so that move tokens are mostly masked
        alpha_t = config.masking_schedule(t).unsqueeze(1).to(device)

        move_mask = torch.rand(moves.size(), device=device) < alpha_t
        masked_moves = torch.where(move_mask, moves, config.mask_token)
        masked_tokens = torch.cat([fens, masked_moves], dim=1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits = model(masked_tokens)
            
        unwrapped_model = model.module if distributed else model
        loss_scale = (config.fen_length + config.move_length) / config.move_length
        return unwrapped_model.elbo_loss(t, logits, tokens, masked_tokens) * loss_scale
    
    def compute_whole_loss(model, fens, moves):
        tokens = torch.cat([fens, moves], dim=1)
        batch_size = len(fens)
        t = ((torch.rand(1) + torch.arange(1, batch_size + 1, 1) / batch_size) % 1).to(device)
        alpha_t = config.masking_schedule(t).unsqueeze(1).to(device)

        mask = torch.rand(tokens.size(), device=device) < alpha_t
        masked_tokens = torch.where(mask, tokens, config.mask_token)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits = model(masked_tokens)
            
        unwrapped_model = model.module if distributed else model
        return unwrapped_model.elbo_loss(t, logits, tokens, masked_tokens)

    # ====================== METRICS HELPERS ======================
    def get_puzzle_local(fen):
        if fen is None or not legal(fen): return None, None
        with SimpleEngine.popen_uci(base_path / ".." / "Stockfish" / "src" / "stockfish") as stockfish:
            puzzle = get_unique_puzzle_from_fen(fen, stockfish)
            best_move = None
            if puzzle:
                best_move = puzzle.mainline[0].move.uci()
            elif legal(fen):
                board = chess.Board(fen)
                info = stockfish.analyse(board, limit=Limit(time=0.1))
                if "pv" in info:
                    best_move = info["pv"][0].uci()
        return puzzle, best_move

    def get_rewards_local(gen_tokens, elbo, step_idx):
        fen_tokens = gen_tokens[:, :config.fen_length]
        move_tokens = gen_tokens[:, config.fen_length:]
        batch_size = len(fen_tokens)
        
        total_length = config.fen_length + config.move_length
        entropies = -elbo / total_length
        
        legal_position = torch.zeros(batch_size, dtype=bool)
        unique_solution = torch.zeros(batch_size, dtype=bool)
        counter_intuitive_solution = torch.zeros(batch_size, dtype=bool)
        piece_counts = torch.zeros(batch_size, dtype=bool)
        correct_move = torch.zeros(batch_size, dtype=bool)
        
        inter_batch_fen_dist = torch.zeros(batch_size, dtype=bool)
        intra_batch_fen_dist = torch.zeros(batch_size, dtype=bool)
        inter_batch_pv_dist = torch.zeros(batch_size, dtype=bool)
        intra_batch_pv_dist = torch.zeros(batch_size, dtype=bool)

        fens = []
        for tokens in fen_tokens:
            try:
                fen = tokens_to_fen(tokens.cpu())
                fens.append(fen)
            except:
                fens.append(None)

        results = list(Parallel(n_jobs=cpu_count)(delayed(get_puzzle_local)(fen) for fen in fens))
        puzzles = [r[0] for r in results]
        best_moves = [r[1] for r in results]

        for i, fen in enumerate(fens):
            if fen is None or not legal(fen): continue
            legal_position[i] = 1
            piece_counts[i] = good_piece_counts(fen)
            counter_intuitive_solution[i] = counter_intuitive(fen, engine)
            
            best_move = best_moves[i]
            if best_move is not None:
                try:
                    predicted_move = tokens_to_move(move_tokens[i].cpu())
                    correct_move[i] = (predicted_move == best_move)
                except:
                    pass

            puzzle = puzzles[i]
            if puzzle is None: continue
            unique_solution[i] = 1
            
            sampled_fens, sampled_pvs, _, _ = buffer.sample(2000)
            pv = " ".join([move.uci() for move in puzzle.mainline])
            intra_batch_fen_dist[i], intra_batch_pv_dist[i] = good_intra_batch_distances(fen, pv, puzzles, i)
            inter_batch_fen_dist[i], inter_batch_pv_dist[i] = good_inter_batch_distances(fen, pv, sampled_fens, sampled_pvs)
        
        index = torch.argmax(unique_solution.float() * counter_intuitive_solution.float() + correct_move.float()).item()
        pred_move = "illegal move"
        try:
            pred_move = tokens_to_move(move_tokens[index].cpu())
        except:
            pass
        validation_writer.add_text(f"Generations", f"fen: {fens[index]} predicted move: {pred_move} best move: {best_moves[index]}", step_idx)
        
        log_data = {
            "entropy": entropies.mean().item(),
            "legal_rate": legal_position.float().mean().item(),
            "uniqueness_rate": unique_solution.float().mean().item(),
            "correct_move_rate": correct_move[legal_position].float().mean().item() if legal_position.any() else 0,
            "counter_intuitive_rate": counter_intuitive_solution.float().mean().item(),
            "unique_and_counter_intuitive": (unique_solution & counter_intuitive_solution).float().mean().item(),
            "piece_counts": piece_counts[legal_position].float().mean().item() if legal_position.any() else 0,
            "dist_inter_fen": inter_batch_fen_dist[unique_solution].float().mean().item() if unique_solution.any() else 0,
            "dist_intra_fen": intra_batch_fen_dist[unique_solution].float().mean().item() if unique_solution.any() else 0,
            "dist_inter_pv": inter_batch_pv_dist[unique_solution].float().mean().item() if unique_solution.any() else 0,
            "dist_intra_pv": intra_batch_pv_dist[unique_solution].float().mean().item() if unique_solution.any() else 0,
        }
        
        for key, value in log_data.items():
            validation_writer.add_scalar(f"Metrics/{key}", value, step_idx)

    # ====================== TRAINING LOOP ======================
    step = checkpoint["step"] if continue_from_checkpoint else 0
    epoch = checkpoint["epoch"] if continue_from_checkpoint else 0
    total = torch.zeros(3, dtype=torch.float32, device=device)

    while step < config.n_steps:
        epoch += 1
        if distributed: trainsampler.set_epoch(epoch)
        
        model.train()
        for fen, move in trainloader:
            step += 1
            fen, move = fen.to(dtype=torch.long, device=device), move.to(dtype=torch.long, device=device)
            
            optimizer.zero_grad()
            
            loss = compute_loss(model, fen, move)
            loss.mean().backward()
            
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            
            total[0] += loss.detach().sum()
            total[1] += len(fen) * norm.detach()
            total[2] += len(fen)

            if step % config.train_logging_interval == 0:
                if distributed: all_reduce(total, op=ReduceOp.SUM)
                if master_process:
                    train_writer.add_scalar("Loss", total[0] / total[2], step)
                    train_writer.add_scalar("Grad norm", total[1] / total[2], step)
                    train_writer.add_scalar("LR", optimizer.param_groups[0]["lr"], step)
                total = torch.zeros(3, dtype=torch.float32, device=device)
                
            if (step % args.validation_interval == 0 or step == 1) and master_process:
                model.eval()
                with torch.no_grad():
                    val_loss_sum = 0
                    val_count = 0
                    for v_fen, v_move in validationloader:
                        v_fen, v_move = v_fen.to(dtype=torch.long, device=device), v_move.to(dtype=torch.long, device=device)
                        val_loss = compute_whole_loss(model, v_fen, v_move).sum()
                        val_loss_sum += val_loss.item()
                        val_count += len(v_fen)
                    validation_writer.add_scalar("Loss", val_loss_sum / val_count, step)

                    # Generation metrics
                    unwrapped_model = model.module if distributed else model
                    gen_tokens = unwrapped_model.sample(batch_size=64, steps=512)
                    
                    # Compute ELBO for entropy monitoring
                    elbo = compute_elbo_basic(model, gen_tokens)
                    
                    get_rewards_local(gen_tokens, elbo, step)
                model.train()

            if step % args.save_interval == 0 and master_process:
                checkpoint_path = logging_path / f"model_{step:07d}.pt"
                save_model = model.module if distributed else model
                torch.save({
                    "model": save_model.state_dict(),
                    "config": config,
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "epoch": epoch
                }, checkpoint_path)

            if step >= config.n_steps: break

    if master_process:
        engine.quit()
        train_writer.close()
        validation_writer.close()
    if distributed: destroy_process_group()

if __name__ == "__main__":
    main()
