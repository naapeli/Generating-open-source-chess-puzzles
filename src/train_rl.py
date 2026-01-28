from pathlib import Path
from copy import deepcopy

import torch
from torch.optim import AdamW, Muon
from chess.engine import SimpleEngine

from MaskedDiffusion.model import MaskedDiffusion
from Config import Config
from rl.espo import espo_loss, generate_grouped_positions, generate_random_themes
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

n_gradient_updates_per_generation = 8  # https://arxiv.org/pdf/2512.03759 figure 5 (24 seems reasonable)
total_steps = 20_000
batch_size = 2
group_size = 8  # 8 - 16 probably good?  # TODO: Look at the deep seek paper where GRPO was proposed (or the Dr GRPO paper)

params_adam = [p for p in model.parameters() if p.ndim != 2]
params_muon = [p for p in model.parameters() if p.ndim == 2]
adam = AdamW(params_adam, lr=1e-3, weight_decay=0)
muon = Muon(params_muon, lr=1e-3, weight_decay=0)

engine = SimpleEngine.popen_uci(base_path / ".." / "Stockfish" / "src" / "stockfish")
# engine.configure({"Hash": 2048})

def get_reward(fen, themes, rating):
    # TODO: add distance metrics and rating dependence
    if legal(fen):
        engine.configure({"Clear Hash": None})
        puzzle = get_unique_puzzle_from_fen(fen, engine)
        if puzzle is None:
            return 0
        is_counter_intuitive = counter_intuitive(fen, engine)
        if not is_counter_intuitive:
            return 0

        generation_themes = cook(puzzle, engine)
        if len(set(generation_themes).intersection(themes)) > len(themes) / 2:
            return 2
        else:
            return 1  # if a puzzle is good, but of wrong type, could still give it a reward
    else:
        return -2


step = 0
end = False
while not end:
    themes, ratings = generate_random_themes(batch_size)
    themes_one_hot = torch.from_numpy(theme_preprocessor.transform(themes)).to(device=device, dtype=torch.float32)
    scaled_ratings = scale_ratings(ratings).to(device=device, dtype=torch.float32)

    # generate the fens from the model
    step_fens, step_themes, step_ratings = generate_grouped_positions(model, themes_one_hot, scaled_ratings, group_size, steps=256)
    print(step_fens)

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
        loss = espo_loss(model, reference_model, step_fens, step_themes, step_ratings, rewards, group_size, eps=0.2, beta=0.0).mean()
        total_loss += loss.item()

        loss.backward()
        adam.step()
        muon.step()

        if step >= total_steps:
            end = True
    
    print("Step: ", step, " Loss: ", total_loss / n_gradient_updates_per_generation, " rewards: ", rewards.mean().item(), flush=True)




engine.quit()
