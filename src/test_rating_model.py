from pathlib import Path

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from RatingModel.model import RatingModel
from tokenization.tokenization import unscale_ratings


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

base_path = Path("./src")
dataset_path = base_path / "dataset" / "validation" / "validationset.pt"
rating_model_path = base_path / "rating_model_checkpoints" / "model_0063000.pt"

checkpoint = torch.load(rating_model_path, map_location="cpu", weights_only=False)

model = RatingModel(checkpoint["config"]).to(device=device)
model.load_state_dict(checkpoint["model"])
model.eval()

dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
dataloader = DataLoader(dataset, batch_size=48)

errors = []
shuffled_errors = []
with torch.no_grad():
    for i, (fen, theme, rating) in enumerate(dataloader):
        print(i, len(dataloader), flush=True)
        fen, theme, rating = fen.to(dtype=torch.long, device=device), theme.to(dtype=torch.float32, device=device), rating.to(dtype=torch.float32, device=device)
        predicted_ratings = model(fen, theme)
        errors.extend(torch.abs(unscale_ratings(predicted_ratings) - unscale_ratings(rating)).tolist())
        theme = theme[torch.randperm(len(rating), device=device)]
        predicted_ratings = model(fen, theme)
        shuffled_errors.extend(torch.abs(unscale_ratings(predicted_ratings) - unscale_ratings(rating)).tolist())

errors = torch.tensor(errors)
shuffled_errors = torch.tensor(shuffled_errors)
print(f"Standard Mean: {errors.mean():.4f}")
print(f"Standard Median: {errors.median():.4f}")
print(f"Shuffled Mean: {shuffled_errors.mean():.4f}")
print(f"Shuffled Median: {shuffled_errors.median():.4f}")

plt.figure()
plt.hist(errors, bins=50, density=True, alpha=0.5, label="Normal Errors")
plt.hist(shuffled_errors, bins=50, density=True, alpha=0.5, label="Shuffled Errors")
plt.xlabel("Rating absolute error")
plt.legend()
plt.savefig("src/rating_model_errors.svg")
