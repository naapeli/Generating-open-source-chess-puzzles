# from pathlib import Path
# from copy import deepcopy
# from datetime import datetime
# import argparse
# import os
# from joblib import Parallel, delayed
# import random
# import io

# import torch
# # Removed optimizers and tensorboard imports as requested
# import chess
# from chess import svg
# from chess.engine import SimpleEngine
# import cairosvg
# from PIL import Image, ImageDraw, ImageFont
# import numpy as np

# # Assuming these imports work in your environment
# from MaskedDiffusion.model import MaskedDiffusion
# from rl.espo import generate_grouped_positions, generate_random_themes, compute_elbo, theme_reward
# from tokenization.tokenization import theme_preprocessor, scale_ratings, tokens_to_fen
# from metrics.themes import legal, get_unique_puzzle_from_fen, counter_intuitive
# from metrics.cook import cook
# from metrics.diversity_filtering import ReplayBuffer
# from metrics.rewards import good_piece_counts

# # ====================== CONFIG ======================
# parser = argparse.ArgumentParser()
# parser.add_argument("--distributed", action="store_true")
# args = parser.parse_args()
# distributed = args.distributed

# base_path = Path("./src")

# # ====================== DEVICE ======================
# rank = 0
# local_rank = 0
# world_size = 1
# master_process = True
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = torch.device(device)

# torch.set_float32_matmul_precision("high")

# # ====================== MODEL LOADING ======================
# checkpoint = torch.load(base_path / "supervised_checkpoints" / "model_0280000.pt", map_location="cpu", weights_only=False)

# config = checkpoint["config"]
# model = MaskedDiffusion(config)
# model.load_state_dict(checkpoint["model"])
# model.to(device=device)

# # Removed Reference Model, Buffers, Optimizers, Schedulers

# cpu_count = int(os.environ.get("SLURM_CPUS_PER_TASK", 4)) 
# engine_path = base_path / ".." / "Stockfish" / "src" / "stockfish"
# engine = SimpleEngine.popen_uci(engine_path)

# # ====================== REWARDS & UTILS ======================

# def get_puzzle(fen):
#     if fen is None: return None
#     if not legal(fen): return None
#     # Spawning separate engines for parallel processing
#     with SimpleEngine.popen_uci(engine_path) as stockfish:
#         puzzle = get_unique_puzzle_from_fen(fen, stockfish)
#     return puzzle

# def get_rewards_and_info(fen_tokens, themes, ratings):
#     """
#     Modified to return detailed info for visualization instead of just the tensor.
#     """
#     legal_position = torch.zeros(len(fen_tokens), dtype=bool)
#     unique_solution = torch.zeros(len(fen_tokens), dtype=bool)
#     counter_intuitive_solution = torch.zeros(len(fen_tokens), dtype=bool)
#     piece_counts = torch.zeros(len(fen_tokens), dtype=bool)
#     themes_match = torch.zeros(len(fen_tokens), dtype=bool)

#     fens = []
#     for tokens in fen_tokens:
#         try:
#             fen = tokens_to_fen(tokens)
#             fens.append(fen)
#         except:
#             fens.append(None)

#     puzzles = list(Parallel(n_jobs=cpu_count)(delayed(get_puzzle)(fen) for fen in fens))

#     group_size = len(fen_tokens) // len(ratings)
    
#     # Store per-item details for the grid
#     results = []

#     for i, fen in enumerate(fens):
#         theme, rating = themes[i // group_size], ratings[i // group_size]
#         info = {
#             "fen": fen,
#             "theme": theme,
#             "rating": rating,
#             "legal": False,
#             "reward": -2.0
#         }

#         if fen is None or not legal(fen):
#             results.append(info)
#             continue
        
#         legal_position[i] = 1
#         info["legal"] = True
        
#         puzzle = puzzles[i]
#         if puzzle is None:
#             # Legal but no puzzle found
#             info["reward"] = 0.0
#             results.append(info)
#             continue
            
#         unique_solution[i] = 1
#         counter_intuitive_solution[i] = counter_intuitive(fen, engine)
#         piece_counts[i] = good_piece_counts(puzzle)

#         generation_themes = cook(puzzle, engine)
#         themes_match[i] = theme_reward(theme, generation_themes)

#         # Calculate individual reward
#         r = 2 * counter_intuitive_solution[i] + 0.5 * piece_counts[i] + 0.5 * themes_match[i]
#         info["reward"] = r.item()
#         results.append(info)

#     return results

# def generate_grid(results, output_file="grid_visualization.png"):
    
#     # Grid settings
#     rows, cols = 4, 4
#     cell_size = 300
#     text_height = 100
#     cell_h = cell_size + text_height
#     grid_img = Image.new("RGB", (cols * cell_size, rows * cell_h), (255, 255, 255))
#     draw_grid = ImageDraw.Draw(grid_img)

#     for idx, info in enumerate(results):
#         if idx >= 16: break # Only take first 16
        
#         row = idx // cols
#         col = idx % cols
#         x_offset = col * cell_size
#         y_offset = row * cell_h

#         # Draw Board
#         if info["fen"]:
#             try:
#                 board = chess.Board(info["fen"])
#                 svg_data = svg.board(board, size=cell_size)
#                 png_data = cairosvg.svg2png(bytestring=svg_data.encode("utf-8"))
#                 board_img = Image.open(io.BytesIO(png_data)).convert("RGB")
#                 grid_img.paste(board_img, (x_offset, y_offset))
#             except Exception as e:
#                 pass
#         else:
#             # Placeholder for invalid FEN
#             draw_grid.rectangle(
#                 [x_offset, y_offset, x_offset + cell_size, y_offset + cell_size], 
#                 fill=(200, 200, 200)
#             )
#             draw_grid.text((x_offset + 10, y_offset + 140), "Invalid FEN", fill=(0,0,0))

#         # Draw Stats Text
#         text_bg = Image.new("RGB", (cell_size, text_height), (240, 240, 240))
#         grid_img.paste(text_bg, (x_offset, y_offset + cell_size))
        
#         d = ImageDraw.Draw(grid_img)
#         theme_txt = f"Theme: {info['theme']}"
#         rating_txt = f"Rating: {int(info['rating'])}"
#         reward_txt = f"Rew: {info['reward']:.2f} | Legal: {info['legal']}"
        
#         # Simple positioning
#         d.text((x_offset + 5, y_offset + cell_size + 5), theme_txt, fill=(0, 0, 0))
#         d.text((x_offset + 5, y_offset + cell_size + 25), rating_txt, fill=(0, 0, 0))
#         d.text((x_offset + 5, y_offset + cell_size + 45), reward_txt, fill=(0, 0, 150))
        
#         if info["fen"]:
#             fen_txt = info["fen"]
#             d.text((x_offset + 5, y_offset + cell_size + 65), fen_txt, fill=(80, 80, 80))

#     grid_img.save(output_file)

# # ====================== EXECUTION ======================

# # Force batch size to 16 for a 4x4 grid
# batch_size = 16 
# group_size = 1 # Simplified group size for visualization purposes
# themes, ratings = generate_random_themes(batch_size)
# ratings = ratings.to(device=device, dtype=torch.float32)
# themes_one_hot = torch.from_numpy(theme_preprocessor.transform(themes)).to(device=device, dtype=torch.float32)
# scaled_ratings = scale_ratings(ratings).to(device=device, dtype=torch.float32)

# # Generating step_fens (tokens)
# step_fens, step_themes, step_ratings = generate_grouped_positions(model, themes_one_hot, scaled_ratings, group_size, steps=128)

# # Extract valid data from the batch
# # Note: generate_grouped_positions returns tokens. We convert them in get_rewards_and_info
# results = get_rewards_and_info(step_fens, themes, ratings.cpu().numpy())

# # Generate the 4x4 Grid
# generate_grid(results, "grid_visualization.png")

# engine.quit()


from pathlib import Path
from copy import deepcopy
from datetime import datetime
import argparse
import os
from joblib import Parallel, delayed
import random
import io

import torch
import chess
from chess import svg
from chess.engine import SimpleEngine
import cairosvg
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Assuming these imports work in your environment
from MaskedDiffusion.model import MaskedDiffusion
from rl.espo import generate_grouped_positions, generate_random_themes, compute_elbo, theme_reward
from tokenization.tokenization import theme_preprocessor, scale_ratings, tokens_to_fen
from metrics.themes import legal, get_unique_puzzle_from_fen, counter_intuitive
from metrics.cook import cook
from metrics.diversity_filtering import ReplayBuffer
from metrics.rewards import good_piece_counts

# ====================== CONFIG ======================
parser = argparse.ArgumentParser()
parser.add_argument("--distributed", action="store_true")
args = parser.parse_args()
distributed = args.distributed

base_path = Path("./src")

# ====================== DEVICE ======================
rank = 0
local_rank = 0
world_size = 1
master_process = True
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

torch.set_float32_matmul_precision("high")

# ====================== MODEL LOADING ======================
checkpoint = torch.load(base_path / "supervised_checkpoints" / "model_0280000.pt", map_location="cpu", weights_only=False)

config = checkpoint["config"]
model = MaskedDiffusion(config)
model.load_state_dict(checkpoint["model"])
model.to(device=device)

cpu_count = int(os.environ.get("SLURM_CPUS_PER_TASK", 4)) 
engine_path = base_path / ".." / "Stockfish" / "src" / "stockfish"
engine = SimpleEngine.popen_uci(engine_path)

# ====================== REWARDS & UTILS ======================

def get_puzzle(fen):
    if fen is None: return None
    if not legal(fen): return None
    # Spawning separate engines for parallel processing
    with SimpleEngine.popen_uci(engine_path) as stockfish:
        puzzle = get_unique_puzzle_from_fen(fen, stockfish)
    return puzzle

def get_rewards_and_info(fen_tokens, themes, ratings):
    """
    Modified to return detailed info for visualization instead of just the tensor.
    """
    legal_position = torch.zeros(len(fen_tokens), dtype=bool)
    unique_solution = torch.zeros(len(fen_tokens), dtype=bool)
    counter_intuitive_solution = torch.zeros(len(fen_tokens), dtype=bool)
    piece_counts = torch.zeros(len(fen_tokens), dtype=bool)
    themes_match = torch.zeros(len(fen_tokens), dtype=bool)

    fens = []
    for tokens in fen_tokens:
        try:
            fen = tokens_to_fen(tokens)
            fens.append(fen)
        except:
            fens.append(None)

    puzzles = list(Parallel(n_jobs=cpu_count)(delayed(get_puzzle)(fen) for fen in fens))

    group_size = len(fen_tokens) // len(ratings)
    
    results = []

    for i, fen in enumerate(fens):
        theme, rating = themes[i // group_size], ratings[i // group_size]
        info = {
            "fen": fen,
            "theme": theme,
            "rating": rating,
            "legal": False,
            "reward": -2.0
        }

        if fen is None or not legal(fen):
            results.append(info)
            continue
        
        legal_position[i] = 1
        info["legal"] = True
        
        puzzle = puzzles[i]
        if puzzle is None:
            # Legal but no puzzle found
            info["reward"] = 0.0
            results.append(info)
            continue
            
        unique_solution[i] = 1
        counter_intuitive_solution[i] = counter_intuitive(fen, engine)
        piece_counts[i] = good_piece_counts(puzzle)

        generation_themes = cook(puzzle, engine)
        themes_match[i] = theme_reward(theme, generation_themes)

        # Calculate individual reward
        r = 2 * counter_intuitive_solution[i] + 0.5 * piece_counts[i] + 0.5 * themes_match[i]
        info["reward"] = r.item()
        results.append(info)

    return results

def generate_grid(results, output_file="grid_visualization.png"):
    
    # Grid settings
    rows, cols = 4, 4
    cell_size = 300
    text_height = 100
    cell_h = cell_size + text_height
    grid_img = Image.new("RGB", (cols * cell_size, rows * cell_h), (255, 255, 255))
    draw_grid = ImageDraw.Draw(grid_img)

    for idx, info in enumerate(results):
        if idx >= 16: break # Only take first 16
        
        row = idx // cols
        col = idx % cols
        x_offset = col * cell_size
        y_offset = row * cell_h

        # Draw Board
        if info["fen"]:
            try:
                board = chess.Board(info["fen"])
                svg_data = svg.board(board, size=cell_size)
                png_data = cairosvg.svg2png(bytestring=svg_data.encode("utf-8"))
                board_img = Image.open(io.BytesIO(png_data)).convert("RGB")
                grid_img.paste(board_img, (x_offset, y_offset))
            except Exception as e:
                pass
        else:
            # Placeholder for invalid FEN
            draw_grid.rectangle(
                [x_offset, y_offset, x_offset + cell_size, y_offset + cell_size], 
                fill=(200, 200, 200)
            )
            draw_grid.text((x_offset + 10, y_offset + 140), "Invalid FEN", fill=(0,0,0))

        # Draw Stats Text
        text_bg = Image.new("RGB", (cell_size, text_height), (240, 240, 240))
        grid_img.paste(text_bg, (x_offset, y_offset + cell_size))
        
        d = ImageDraw.Draw(grid_img)
        theme_txt = f"Theme: {info['theme']}"
        rating_txt = f"Rating: {int(info['rating'])}"
        reward_txt = f"Rew: {info['reward']:.2f} | Legal: {info['legal']}"
        
        # Simple positioning
        d.text((x_offset + 5, y_offset + cell_size + 5), theme_txt, fill=(0, 0, 0))
        d.text((x_offset + 5, y_offset + cell_size + 25), rating_txt, fill=(0, 0, 0))
        d.text((x_offset + 5, y_offset + cell_size + 45), reward_txt, fill=(0, 0, 150))
        
        if info["fen"]:
            fen_txt = info["fen"]
            d.text((x_offset + 5, y_offset + cell_size + 65), fen_txt, fill=(80, 80, 80))

    grid_img.save(output_file)

# ====================== EXECUTION ======================

# We want exactly 16 positions with nonzero reward.
target_count = 16
accumulated_results = []
batch_size = 16 # Batch size for generation
group_size = 1 


while len(accumulated_results) < target_count:
    themes, ratings = generate_random_themes(batch_size)
    ratings = ratings.to(device=device, dtype=torch.float32)
    themes_one_hot = torch.from_numpy(theme_preprocessor.transform(themes)).to(device=device, dtype=torch.float32)
    scaled_ratings = scale_ratings(ratings).to(device=device, dtype=torch.float32)

    # Generating step_fens (tokens)
    step_fens, step_themes, step_ratings = generate_grouped_positions(model, themes_one_hot, scaled_ratings, group_size, steps=128)

    # Get results for this batch
    batch_results = get_rewards_and_info(step_fens, themes, ratings.cpu().numpy())

    # Filter for nonzero rewards
    # Note: Reward 0.0 usually means legal but no puzzle. 
    # Reward -2.0 is illegal (also nonzero). Reward > 0 is a puzzle.
    # Assuming 'nonzero' implies strictly filtering out the 0.0 "failures".
    for res in batch_results:
        if abs(res['reward']) > 1e-6:
            accumulated_results.append(res)
            # Stop immediately if we filled the quota
            if len(accumulated_results) >= target_count:
                break
    
# Generate the 4x4 Grid with the filtered results
generate_grid(accumulated_results[:16], "grid_visualization.png")

engine.quit()
