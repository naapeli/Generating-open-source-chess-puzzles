from pathlib import Path

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from RatingModel.model import RatingModel
from tokenization.tokenization import unscale_ratings


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

base_path = Path("./src")
dataset_path = base_path / "dataset" / "validation" / "validationset.pt"
rating_model_path = base_path / "rating_model_checkpoints" / "model_0010500.pt"

checkpoint = torch.load(rating_model_path, map_location="cpu", weights_only=False)

model = RatingModel(checkpoint["config"]).to(device=device)
model.load_state_dict(checkpoint["model"])


dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
dataloader = DataLoader(dataset, batch_size=48)

errors = []
for i, (fen, theme, rating) in enumerate(dataloader):
    print(i, len(dataloader), flush=True)
    fen, theme, rating = fen.to(dtype=torch.long, device=device), theme.to(dtype=torch.float32, device=device), rating.to(dtype=torch.float32, device=device)
    predicted_ratings = model(fen, theme)
    errors.extend(torch.abs(unscale_ratings(predicted_ratings) - unscale_ratings(rating)).tolist())

errors = torch.tensor(errors)
print(errors.mean())
print(errors.median())

plt.hist(errors, bins=50, density=True)
plt.xlabel("Rating absolute error")
plt.savefig("src/rating_model_errors.svg")
