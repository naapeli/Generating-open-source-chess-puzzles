import torch
from stockfish import Stockfish

from pathlib import Path

from MaskedDiffusion.model import MaskedDiffusion
from metrics import legal
from tokenization.tokenization import tokens_to_fen, scale_ratings, get_themes


stockfish = Stockfish(path="./Stockfish/src/stockfish", depth=15)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

base_path = Path("./src")
checkpoint = torch.load(base_path / "supervised_checkpoints" / "model_0280000.pt", map_location="cpu", weights_only=False)

config = checkpoint["config"]
model = MaskedDiffusion(config)
model.load_state_dict(checkpoint["model"])
model.to(device=device)
model.eval()

N_GENERATIONS = 1_000
N_GENERATIONS_PER_STEP = 256
step = 0
ended = False
while not ended:
    themes = torch.zeros((N_GENERATIONS_PER_STEP, config.n_themes), dtype=torch.float32, device=device)
    indices = torch.randint(0, config.n_themes, (N_GENERATIONS_PER_STEP,))
    rows = torch.arange(N_GENERATIONS_PER_STEP, device=device)
    themes[rows, indices] = 1
    ratings = 3000 * torch.rand(N_GENERATIONS_PER_STEP, dtype=torch.float32, device=device) + 300
    scaled_ratings = scale_ratings(ratings)
    fens = model.sample(themes, scaled_ratings, steps=128)
    theme_strings = get_themes(themes.cpu().numpy())
    for generated_fen, rating, theme_string in zip(fens, ratings, theme_strings):
        step += 1
        try:
            fen = tokens_to_fen(generated_fen)
            string = str(step) + "," + fen + "," + str(int(rating.item())) + "," + theme_string[0] + "," + str(legal(stockfish, fen)) + "\n"
            with open(base_path / "generated_fens.txt", "a") as f:
                f.write(string)
        except KeyError:
            with open(base_path / "generated_fens.txt", "a") as f:
                f.write(str(step) + ",Error," + str(int(rating.item())) + "," + theme_string[0] + ",False\n")
        
        if step >= N_GENERATIONS:
            ended = True
            break
