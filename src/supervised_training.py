import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW, Muon
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from MaskedDiffusion.model import MaskedDiffusion
from MaskingSchedule.MaskingSchedule import LinearSchedule, PolynomialSchedule, GeometricSchedule, CosineSchedule
from Config import Config
from tokenization.tokenization import tokens_to_fen


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
config = Config(n_layers=1, batch_size=32, n_steps=100, validation_interval=25, n_validation_generations=1, embed_dim=128)

base_path = Path("./src/dataset")
fen_tokens = torch.load(base_path / "fen_tokens.pt")
theme_tokens = torch.load(base_path / "theme_tokens.pt")
ratings = torch.load(base_path / "ratings.pt")
dataset = TensorDataset(fen_tokens, theme_tokens, ratings)
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
validation_fen, validation_theme, validation_rating = next(iter(dataloader))  # TODO: in the future, implement a real validation set

model = MaskedDiffusion(config)
model.to(device=device, dtype=torch.float32)

params_adam = [p for p in model.parameters() if p.ndim != 2]
params_muon = [p for p in model.parameters() if p.ndim == 2]
adam = AdamW(params_adam, lr=config.lr, weight_decay=config.weight_decay)
muon = Muon(params_muon, lr=config.lr, weight_decay=config.weight_decay)

validation_log_probs = []
train_log_probs = []
losses = []
steps = 0
ended = False
while not ended:
    for fen, theme, rating in dataloader:
        fen, theme, rating = fen.to(dtype=torch.long, device=device), theme.to(dtype=torch.float32, device=device), rating.to(dtype=torch.float32, device=device)

        batch_size = len(fen)
        t = (torch.rand(1) + torch.arange(1, batch_size + 1, 1) / batch_size) % 1
        alpha_t = config.masking_schedule(t).unsqueeze(1)
        random_mask = torch.rand(fen.size()) < alpha_t
        masked_fen = torch.where(random_mask, fen, config.mask_token)

        adam.zero_grad()
        muon.zero_grad()

        logits = model(masked_fen, theme, rating)
        loss = model.elbo_loss(t, logits, fen, masked_fen)
        
        loss.backward()
        adam.step()
        muon.step()

        losses.append(loss.item())
        train_log_probs.append(model.log_prob(fen, theme, rating).sum().item())
        validation_log_probs.append(model.log_prob(validation_fen.to(torch.long), validation_theme.to(torch.float32), validation_rating.to(torch.float32)).sum().item())  # TODO: do something better

        steps += 1

        print(f"Steps: {steps} - Train loss: {loss.item()}")

        if steps > config.n_steps:
            ended = True
            break

        if steps % config.validation_interval == 0:
            validation_themes = torch.zeros((config.n_validation_generations, config.n_themes), dtype=torch.float32)
            validation_themes[:, torch.randint(0, config.n_themes, (config.n_validation_generations,))] = 1
            validation_ratings = 3000 * torch.rand(config.n_validation_generations, dtype=torch.float32)
            # temp_fen = torch.full((config.n_validation_generations, config.fen_length), config.mask_token, device=device, dtype=torch.long)
            # logits = model(temp_fen, validation_themes, validation_ratings)
            # print(logits[0, 0], logits[0, 15], logits[0, 70])
            fens = model.sample(validation_themes, validation_ratings, steps=5)
            print(fens.squeeze().numpy())
            for generated_fen in fens:
                try:
                    print(tokens_to_fen(generated_fen))
                except:
                    pass


plt.subplot(121)
losses = np.array(losses)
window_size = 10
window = np.ones(window_size) / window_size
moving_average = np.convolve(losses, window, mode="valid")
plt.plot(np.arange(window_size - 1, len(losses)), moving_average, label="Moving Average")
plt.plot(losses, alpha=0.3, label="Raw Loss")
plt.legend()
plt.xlabel("Steps")
plt.ylabel("Train loss")

plt.subplot(122)
plt.plot(validation_log_probs, label="Validation")
plt.plot(train_log_probs, label="Train")
plt.legend()
plt.xlabel("Steps")
plt.ylabel("Validation log probs")

plt.show()
