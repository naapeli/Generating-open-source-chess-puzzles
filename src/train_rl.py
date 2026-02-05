from pathlib import Path
from copy import deepcopy

import torch
from torch.optim import AdamW, Muon
import chess
from chess.engine import SimpleEngine

from MaskedDiffusion.model import MaskedDiffusion
from rl.espo import espo_loss, generate_grouped_positions, generate_random_themes, compute_elbo, compute_elbo_basic
from tokenization.tokenization import theme_preprocessor, scale_ratings, tokens_to_fen
from metrics.themes import legal, get_unique_puzzle_from_fen, counter_intuitive
from metrics.cook import cook
from metrics.diversity_filtering import ReplayBuffer, PV_distance, board_distance


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


base_path = Path("./src")
checkpoint = torch.load(base_path / "supervised_checkpoints" / "model_0280000.pt", map_location="cpu", weights_only=False)

config = checkpoint["config"]
model = MaskedDiffusion(config)
model.load_state_dict(checkpoint["model"])
model.to(device=device)

reference_model = deepcopy(model)
reference_model.to(device=device)

n_gradient_updates_per_generation = 8  # https://arxiv.org/pdf/2512.03759 figure 5 (8 - 24 seems reasonable)
total_steps = 20_000
batch_size = 4
group_size = 16  # 8 - 64 probably good?  # TODO: Look at the deep seek paper where GRPO was proposed (or the Dr GRPO paper)  in the overfitting task, 16 seems to be too low

params_adam = [p for p in model.parameters() if p.ndim != 2]
params_muon = [p for p in model.parameters() if p.ndim == 2]
adam = AdamW(params_adam, lr=4e-3, weight_decay=0)  # 4e-3
muon = Muon(params_muon, lr=4e-3, weight_decay=0)  # 4e-3

engine = SimpleEngine.popen_uci(base_path / ".." / "Stockfish" / "src" / "stockfish")
# engine.configure({"Hash": 2048})

piece_counts = {chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2, chess.ROOK: 2, chess.QUEEN: 1, chess.KING: 1}
pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
colors = [chess.WHITE, chess.BLACK]

def get_reward(fen, themes, rating):
    # TODO: add distance metrics and rating dependence
    if not legal(fen):
        return -2
    
    engine.configure({"Clear Hash": None})
    puzzle = get_unique_puzzle_from_fen(fen, engine)
    if puzzle is None:
        return 0
    is_counter_intuitive = counter_intuitive(fen, engine)
    if not is_counter_intuitive:
        return 0
    
    # realistic piece counts
    for color in colors:
        for piece in pieces:
            if len(puzzle.game.board().pieces(piece, color)) > piece_counts[piece]:
                return 0
    
    # check the PV distances of the solution and the board
    # TODO

    generation_themes = cook(puzzle, engine)
    if len(set(generation_themes).intersection(themes)) > len(themes) / 2:
        return 2
    else:
        return 1  # if a puzzle is good, but of wrong type, could still give it a reward


step = 0
end = False
while not end:
    themes, ratings = generate_random_themes(batch_size)
    # themes_one_hot = torch.from_numpy(theme_preprocessor.transform(themes)).to(device=device, dtype=torch.float32)
    # scaled_ratings = scale_ratings(ratings).to(device=device, dtype=torch.float32)
    themes_one_hot = torch.from_numpy(theme_preprocessor.transform(themes)).to(device=device, dtype=torch.float32)
    scaled_ratings = scale_ratings(ratings).to(device=device, dtype=torch.float32)
    # model.to(torch.float64)
    # reference_model.to(torch.float64)

    # generate the fens from the old_model
    old_model = deepcopy(model)
    old_model.to(device)
    step_fens, step_themes, step_ratings = generate_grouped_positions(old_model, themes_one_hot, scaled_ratings, group_size, steps=128)  # 256, 32

    # compute the rewards
    rewards = torch.zeros(len(step_fens))
    for i, tokens in enumerate(step_fens):
        group_index = i // group_size
        theme = themes[group_index]
        rating = ratings[group_index]
        # print(tokens, theme, rating)
        # fen = tokens_to_fen(tokens)
        # rewards[i] = get_reward(fen, theme, rating)
        rewards[i] = -(tokens != 0).float().sum()  # model should maximize the amount of zeros

    # update the model many times for one generation (as generations and reward calculations are expensive)
    total_loss = 0
    total_norm = 0
    for substep in range(n_gradient_updates_per_generation):
        step += 1

        adam.zero_grad()
        muon.zero_grad()

        # precompute the elbos of the old model
        with torch.no_grad():
            reference_elbo, mask = compute_elbo(reference_model, step_fens, step_themes, step_ratings, return_mask=True)
            old_elbo = compute_elbo(old_model, step_fens, step_themes, step_ratings, mask=mask)

        loss = espo_loss(model, reference_elbo, old_elbo, step_fens, step_themes, step_ratings, rewards, group_size, mask, eps=0.999, beta=1e-4).mean()
        total_loss += loss.item()

        loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        total_norm += norm.item()

        adam.step()
        muon.step()

        if step >= total_steps:
            end = True
    
    print("Step: ", step, " Loss: ", total_loss / n_gradient_updates_per_generation, " reward: ", rewards.mean().item(), " grad norm: ", total_norm / n_gradient_updates_per_generation, flush=True)


engine.quit()
