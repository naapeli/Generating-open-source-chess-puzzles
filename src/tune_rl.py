from pathlib import Path
from copy import deepcopy
import argparse
import os
from joblib import Parallel, delayed
import random
import io
import gc
import numpy as np

import optuna
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.tensorboard import SummaryWriter
import chess
from chess import svg
from chess.engine import SimpleEngine, Limit
import cairosvg
from PIL import Image, ImageDraw

from MaskedDiffusion.model import MaskedDiffusion
from RatingModel.model import RatingModel
from rl.espo import espo_loss, generate_grouped_positions, generate_random_themes, compute_elbo, theme_reward, critic_free_ppo_loss, entropy
from tokenization.tokenization import theme_preprocessor, scale_ratings, unscale_ratings, tokens_to_fen, tokenize_fen, tokens_to_move
from metrics.themes import legal, get_unique_puzzle_from_fen, counter_intuitive
from metrics.cook import cook
from metrics.diversity_filtering import ReplayBuffer
from metrics.rewards import good_piece_counts, good_inter_batch_distances, good_intra_batch_distances


def get_stockfish_data(fen, model_move, engine_path):
    if fen is None: return None, None, None
    if not legal(fen): return None, None, None
    with SimpleEngine.popen_uci(engine_path) as stockfish:
        board = chess.Board(fen)
        limit = Limit(depth=15, time=10, nodes=8_000_000)
        
        analysis = stockfish.analyse(board, limit=limit)
        best_move = analysis["pv"][0] if "pv" in analysis else None
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
            return puzzle, None, cp_loss
    return puzzle, best_move.uci(), cp_loss

def objective(trial):
    # Hyperparameters from trial
    lr = trial.suggest_float("lr", 1e-8, 1e-2, log=True)
    beta = trial.suggest_float("beta", 1e-8, 10.0, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1.e-2, log=True)
    
    # Static parameters
    n_gradient_updates_per_generation = 1
    total_steps = args.steps_per_trial
    batch_size = 2  # 16
    local_batch_size = batch_size
    group_size = 32  # 4
    eps = 0.3
    lichess_distribution = True
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    device = torch.device(device)

    # Seed
    seed = 42 + trial.number
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_float32_matmul_precision("high")

    # Logging
    logging_path = base_path / "runs"/ "rl" / "tune" / args.study_name / f"trial_{trial.number}"
    writer = SummaryWriter(logging_path)

    reference_checkpoint_path = base_path / "runs" / "supervised" / args.model_path
    reference_checkpoint = torch.load(reference_checkpoint_path, map_location="cpu", weights_only=False)
    
    config = deepcopy(reference_checkpoint["config"])
    config.lr = lr
    config.weight_decay = weight_decay
    
    model = MaskedDiffusion(config)
    model.load_state_dict(reference_checkpoint["model"])
    model.to(device=device)

    rating_model_checkpoint = torch.load(base_path / "runs" / "rating_model" / "v1" / "model_0063000.pt", map_location="cpu", weights_only=False)
    rating_model = RatingModel(rating_model_checkpoint["config"])
    rating_model.load_state_dict(rating_model_checkpoint["model"])
    rating_model.to(device=device)
    
    reference_model = MaskedDiffusion(reference_checkpoint["config"])
    reference_model.load_state_dict(reference_checkpoint["model"])
    reference_model.to(device=device)
    
    capacity = 200_000
    buffer = ReplayBuffer(capacity, base_path / "dataset" / "rl")

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    lr_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=50, last_epoch=-1)
    
    cpu_count = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    
    engine_path = base_path / ".." / "Stockfish" / "src" / "stockfish"
    engine = SimpleEngine.popen_uci(engine_path)

    def save_board(fen, themes, rating, tag, step):
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

    def get_rewards_local(fen_tokens, theme_tokens, ratings, elbo, step):
        legal_position = torch.zeros(len(fen_tokens), dtype=bool)
        unique_solution = torch.zeros(len(fen_tokens), dtype=bool)
        counter_intuitive_solution = torch.zeros(len(fen_tokens), dtype=bool)
        piece_counts = torch.zeros(len(fen_tokens), dtype=bool)
        inter_batch_fen_dist = torch.zeros(len(fen_tokens), dtype=bool)
        intra_batch_fen_dist = torch.zeros(len(fen_tokens), dtype=bool)
        inter_batch_pv_dist = torch.zeros(len(fen_tokens), dtype=bool)
        intra_batch_pv_dist = torch.zeros(len(fen_tokens), dtype=bool)
        themes_match = torch.zeros(len(fen_tokens), dtype=bool)
        move_matches = torch.zeros(len(fen_tokens), dtype=bool)
        cp_losses = torch.full((len(fen_tokens),), float("nan"), dtype=torch.float32)
        rating_penalty = torch.zeros(len(fen_tokens), dtype=torch.float32)

        _, sequence_length = fen_tokens.shape
        entropies = entropy(elbo.cpu(), sequence_length)

        if config.use_context:
            group_size = len(fen_tokens) // len(ratings)
            themes = theme_preprocessor.inverse_transform(theme_tokens.cpu().numpy())
            true_ratings = ratings.repeat_interleave(group_size, dim=0)

        device = fen_tokens.device
        fen_tokens_cpu = fen_tokens.cpu()

        fens = []
        model_moves = []
        for tokens in fen_tokens_cpu:
            try:
                fen = tokens_to_fen(tokens[:config.fen_length])
                fens.append(fen)
            except:
                fens.append(None)
            try:
                model_move = tokens_to_move(tokens[config.fen_length:])
                model_moves.append(model_move)
            except:
                model_moves.append(None)

        try:
            results = list(Parallel(n_jobs=cpu_count)(delayed(get_stockfish_data)(fen, model_moves[i], engine_path) for i, fen in enumerate(fens)))
            puzzles = [r[0] for r in results]
            actual_best_moves = [r[1] for r in results]
            found_cp_losses = [r[2] for r in results]
        except TimeoutError:
            return torch.zeros(len(fen_tokens), dtype=torch.float32)

        valid_indices = []
        valid_gen_themes = []

        for i, fen in enumerate(fens):
            if config.use_context:
                theme = themes[i // group_size]
            if fen is None or not legal(fen):
                continue
            legal_position[i] = 1

            piece_counts[i] = good_piece_counts(fen)
            
            actual_best_move = actual_best_moves[i]
            move_matches[i] = (model_moves[i] == actual_best_move)
            if found_cp_losses[i] is not None:
                cp_losses[i] = found_cp_losses[i]

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
            if config.use_context:
                themes_match[i] = theme_reward(theme, generation_themes)

            valid_indices.append(i)
            valid_gen_themes.append(generation_themes)

        if valid_indices and config.use_context:
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
        
        pass_diversity_filtering = intra_distances & inter_batch_fen_dist & piece_counts
        rewards = torch.where(legal_position & pass_diversity_filtering & unique_solution & counter_intuitive_solution & move_matches, 10, 0)
        rewards = torch.where(~legal_position, -1, rewards)

        rewards = rewards.float()
        log_rewards = rewards.clone()
        if log_rewards.numel() > 0:
            index = torch.argmax(log_rewards).item()
            board_theme = themes[index // group_size] if config.use_context else None
            board_rating = ratings[index // group_size].item() if config.use_context else None
            save_board(fens[index], board_theme, board_rating, "Generations", step)

        is_valid = torch.zeros(len(fen_tokens), dtype=torch.bool)
        is_valid[valid_indices] = True
        valid_mask = is_valid

        log_data = {}
        log_data["legal_rate"] = legal_position.float().mean().item()
        log_data["uniqueness_rate"] = unique_solution.float().mean().item()
        log_data["counter_intuitive_rate"] = counter_intuitive_solution.float().mean().item()
        log_data["entropy"] = entropies.float().mean().item()

        if (legal_position).any():
            log_data["move_match_rate"] = move_matches[legal_position].float().mean().item()
        if (valid_mask & ~torch.isnan(cp_losses)).any():
            log_data["cp_loss"] = cp_losses[valid_mask & ~torch.isnan(cp_losses)].mean().item()

        if valid_mask.any():
            log_data["counter_intuitive_rate_given_unique"] = counter_intuitive_solution[valid_mask].float().mean().item()
            log_data["themes_match_rate"] = themes_match[valid_mask].float().mean().item()
            log_data["dist_inter_fen"] = inter_batch_fen_dist[valid_mask].float().mean().item()
            log_data["dist_intra_fen"] = intra_batch_fen_dist[valid_mask].float().mean().item()
            log_data["dist_inter_pv"] = inter_batch_pv_dist[valid_mask].float().mean().item()
            log_data["dist_intra_pv"] = intra_batch_pv_dist[valid_mask].float().mean().item()
            log_data["intra_dist"] = intra_distances[valid_mask].float().mean().item()
            log_data["inter_dist"] = inter_distances[valid_mask].float().mean().item()
            log_data["all_dist"] = all_distances[valid_mask].float().mean().item()
            log_data["rating_abs_diff"] = (-1000 * rating_penalty[valid_mask]).float().mean().item()
            log_data["pass_diversity_filtering"] = pass_diversity_filtering[valid_mask].float().mean().item()

        if (legal_position).any():
            log_data["piece_counts"] = piece_counts[legal_position].float().mean().item()
        
        for key, value in log_data.items():
            writer.add_scalar(f"Components/{key}", value, step)
        writer.add_scalar("Reward", rewards.float().mean().item(), step)

        return rewards.to(device=device, dtype=torch.float32)

    step = 0
    end = False
    
    last_rewards = []
    
    while not end:
        if config.use_context:
            themes, ratings = generate_random_themes(local_batch_size, lichess_distribution=lichess_distribution)
            ratings = ratings.to(device=device, dtype=torch.float32)
            themes_one_hot = torch.from_numpy(theme_preprocessor.transform(themes)).to(device=device, dtype=torch.float32)
            scaled_ratings = scale_ratings(ratings).to(device=device, dtype=torch.float32)
        else:
            themes = None
            ratings = None
            themes_one_hot = None
            scaled_ratings = None

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            step_fens, step_themes, step_ratings = generate_grouped_positions(model, themes_one_hot, scaled_ratings, group_size, batch_size, steps=512)
        
        with torch.no_grad():
            reference_elbo, mask, t = compute_elbo(reference_model, step_fens, step_themes, step_ratings, return_mask=True)
            old_elbo = compute_elbo(model, step_fens, step_themes, step_ratings, mask=mask)

        rewards = get_rewards_local(step_fens, themes_one_hot, ratings, old_elbo, step)
        
        mean_reward = rewards.float().mean().item()
        last_rewards.append(mean_reward)
        if len(last_rewards) > 20:
            last_rewards.pop(0)

        total_loss = 0
        total_norm = 0
        total_kl = 0
        total_clips = 0
        for substep in range(n_gradient_updates_per_generation):
            step += 1

            optimizer.zero_grad()

            loss, kl, is_clipped = espo_loss(model, reference_elbo, old_elbo, step_fens, step_themes, step_ratings, rewards, group_size, mask=mask, t=t, eps=eps, beta=beta)
            loss = loss.mean()
            total_loss += loss
            total_kl += kl.mean()
            total_clips += is_clipped.float().mean()

            loss.backward()

            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            total_norm += norm

            optimizer.step()
            lr_scheduler.step()

        # Let optuna know the current state (we can report after a generation is fully done)
        trial.report(mean_reward, step)
        if trial.should_prune():
            writer.add_hparams({"lr": lr, "beta": beta, "weight_decay": weight_decay}, {"rewards": np.mean(last_rewards) if last_rewards else mean_reward})
            engine.quit()
            writer.close()
            raise optuna.exceptions.TrialPruned()

        if step >= total_steps:
            end = True
        
        writer.add_scalar("Loss/Loss", total_loss.item() / n_gradient_updates_per_generation, step)
        writer.add_scalar("Loss/Grad norm", total_norm.item() / n_gradient_updates_per_generation, step)
        writer.add_scalar("Loss/KL divergence", total_kl.item() / n_gradient_updates_per_generation, step)
        writer.add_scalar("Loss/learning_rate", optimizer.param_groups[0]["lr"], step)
        writer.add_scalar("Loss/Clips", total_clips.item() / n_gradient_updates_per_generation, step)

        torch.cuda.empty_cache()
        gc.collect()

    final_obj = np.mean(last_rewards) if last_rewards else mean_reward

    writer.add_hparams({"lr": lr, "beta": beta, "weight_decay": weight_decay}, {"rewards": final_obj})

    engine.quit()
    writer.close()
    
    return final_obj

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--study_name", type=str, required=True)
    parser.add_argument("--steps_per_trial", type=int, default=2000)
    parser.add_argument("--reward_structure", type=str, default="reward = ")
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    # Disable Optuna's default print logs to keep the console clean
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    base_path = Path("./src")

    tune_dir = base_path / "runs" / "rl" / "tune" / args.study_name
    tune_dir.mkdir(parents=True, exist_ok=True)
    db_path = tune_dir / f"{args.study_name}.db"

    with open(tune_dir / "notes.txt", "w") as f:
        f.write(args.reward_structure)

    # Optuna study with SQLite storage to save progress automatically
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=100)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{db_path.as_posix()}",
        load_if_exists=True,
        direction="maximize", 
        pruner=pruner
    )
    study.enqueue_trial({"le": 1e-5, "beta": 1, "weight_decay": 1e-5})
    study.optimize(objective, n_trials=args.n_trials)
    
    summary_path = tune_dir / f"{args.study_name}_summary.txt"
    
    with open(summary_path, "w") as f:
        f.write("Study statistics:\n")
        f.write(f"  Number of finished trials: {len(study.trials)}\n")
        f.write(f"  Number of pruned trials: {len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED]))}\n")
        f.write(f"  Number of complete trials: {len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]))}\n\n")
        
        try:
            trial = study.best_trial
            f.write("Best trial:\n")
            f.write(f"  Value: {trial.value}\n")
            f.write("  Params:\n")
            for key, value in trial.params.items():
                f.write(f"    {key}: {value}\n")
        except ValueError:
            f.write("  No trials completed successfully.\n")
