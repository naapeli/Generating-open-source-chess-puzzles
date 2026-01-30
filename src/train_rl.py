from pathlib import Path
from copy import deepcopy

import torch
from torch.optim import AdamW, Muon
import chess
from chess.engine import SimpleEngine
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from MaskedDiffusion.model import MaskedDiffusion
from rl.espo import espo_loss, espo_loss_basic, generate_grouped_positions, generate_random_themes
from tokenization.tokenization import theme_preprocessor, scale_ratings, tokens_to_fen
from metrics.themes import legal, get_unique_puzzle_from_fen, counter_intuitive
from metrics.cook import cook


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
batch_size = 1
group_size = 8  # 24  # 8 - 64 probably good?  # TODO: Look at the deep seek paper where GRPO was proposed (or the Dr GRPO paper)  in the overfitting task, 16 seems to be too low

params_adam = [p for p in model.parameters() if p.ndim != 2]
params_muon = [p for p in model.parameters() if p.ndim == 2]
adam = AdamW(params_adam, lr=1e-2, weight_decay=0)  # 4e-3
muon = Muon(params_muon, lr=1e-2, weight_decay=0)  # 4e-3

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
# end = False
end = True
while not end:
    themes, ratings = generate_random_themes(batch_size)
    themes_one_hot = torch.from_numpy(theme_preprocessor.transform(themes)).to(device=device, dtype=torch.float32)
    scaled_ratings = scale_ratings(ratings).to(device=device, dtype=torch.float32)

    # generate the fens from the model
    step_fens, step_themes, step_ratings = generate_grouped_positions(model, themes_one_hot, scaled_ratings, group_size, steps=128)  # 256

    # compute the rewards
    rewards = torch.zeros(len(step_fens))
    for i, (tokens, theme, rating) in enumerate(zip(step_fens, themes, ratings)):
        # fen = tokens_to_fen(tokens)
        # rewards[i] = get_reward(fen, theme, rating)
        rewards[i] = -(tokens != 0).float().sum()  # model should maximize the amount of zeros

    # update the model many times for one generation (as generations and reward calculations are expensive)
    total_loss = 0
    for substep in range(n_gradient_updates_per_generation):
        step += 1

        adam.zero_grad()
        muon.zero_grad()

        loss = espo_loss(model, reference_model, step_fens, step_themes, step_ratings, rewards, group_size, eps=0.1, beta=0.1).mean()
        total_loss += loss.item()

        loss.backward()

        # norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 10
        norm = torch.nn.utils.get_total_norm(model.parameters())

        adam.step()
        muon.step()

        # end = True

        if step >= total_steps:
            end = True
    
    print("Step: ", step, " Loss: ", total_loss / n_gradient_updates_per_generation, " reward: ", rewards.mean().item(), " grad norm: ", norm.item(), flush=True)



batch_size = 10
group_size = 1
themes, ratings = generate_random_themes(batch_size)
themes_one_hot = torch.from_numpy(theme_preprocessor.transform(themes)).to(device=device, dtype=torch.float32)
scaled_ratings = scale_ratings(ratings).to(device=device, dtype=torch.float32)

step_fens, step_themes, step_ratings = generate_grouped_positions(model, themes_one_hot, scaled_ratings, group_size, steps=128)

rewards = torch.randn(len(step_fens))

losses = []
basic_losses = []
beta = 0
eps = 0.1
for substep in range(100):
    loss = espo_loss(model, reference_model, step_fens, step_themes, step_ratings, rewards, group_size, eps=eps, beta=beta)
    losses.extend(loss.tolist())
    
    loss = espo_loss_basic(model, reference_model, step_fens, step_themes, step_ratings, rewards, group_size, eps=eps, beta=beta)
    basic_losses.extend(loss.tolist())

losses = np.array(losses)
basic_losses = np.array(basic_losses)


def plot_with_mode(data, label, color, x_limit=None):
    kde = gaussian_kde(data)
    x_limit = data.max() if x_limit is None else x_limit
    x_range = np.linspace(0, x_limit, 1000)
    pdf_values = kde(x_range)
    mode_x = x_range[np.argmax(pdf_values)]
    plt.plot(x_range, pdf_values, color=color, lw=2, label=f"{label} PDF")
    plt.axvline(mode_x, color=color, linestyle='--', alpha=0.7, 
                label=f"{label} Mode: {mode_x:.2f}")
    plt.fill_between(x_range, pdf_values, alpha=0.2, color=color)
    return mode_x


plt.figure()
mode_low = plot_with_mode(losses, "Low Var (Quadrature)", "blue")
mode_high = plot_with_mode(basic_losses, "High Var (Basic)", "orange")
plt.legend()
plt.xlim((min(losses.min(), basic_losses.min()) - 2, max(np.quantile(losses, 0.95), np.quantile(basic_losses, 0.95)) + 2))
plt.savefig(base_path / "image.png")


print(mode_low, losses.mean(), np.median(losses), losses.var())
print(mode_high, basic_losses.mean(), np.median(basic_losses), basic_losses.var())


engine.quit()
