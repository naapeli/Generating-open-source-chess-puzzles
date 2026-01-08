import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW, Muon
import matplotlib.pyplot as plt

from pathlib import Path

from MaskedDiffusion.model import MaskedDiffusion
from MaskingSchedule.MaskingSchedule import LinearSchedule, PolynomialSchedule, GeometricSchedule, CosineSchedule
from Config import Config
from tokenization.tokenization import tokens_to_fen, FENTokens


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
config = Config(n_layers=1, batch_size=3, n_steps=100, validation_interval=10, n_validation_generations=1)

base_path = Path("./dataset")
fen_tokens = torch.load(base_path / "fen_tokens.pt")
theme_tokens = torch.load(base_path / "theme_tokens.pt")
ratings = torch.load(base_path / "ratings.pt")
dataset = TensorDataset(fen_tokens, theme_tokens, ratings)
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
validation_fen, validation_theme, validation_rating = next(iter(dataloader))  # TODO: in the future, implement a real validation set

model = MaskedDiffusion(config)
model.to(device=device, dtype=torch.float32)

optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
# optimizer = Muon(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

log_probs = []
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

        optimizer.zero_grad()

        logits = model(masked_fen, theme, rating)
        loss = model.elbo_loss(t, logits, fen, masked_fen)
        
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        log_probs.append(model.log_prob(validation_fen.to(torch.long), validation_theme.to(torch.float32), validation_rating.to(torch.float32)).sum().item())  # TODO: do something better
        
        steps += 1

        if steps > config.n_steps:
            ended = True
            break

        if steps % config.validation_interval == 0:
            print("Steps: ", steps)
            validation_themes = torch.zeros((config.n_validation_generations, config.n_themes), dtype=torch.float32)
            validation_themes[:, torch.randint(0, config.n_themes, (config.n_validation_generations,))] = 1
            validation_ratings = 3000 * torch.rand(config.n_validation_generations, dtype=torch.float32)
            fens = model.sample(validation_themes, validation_ratings, steps=5)
            for generated_fen in fens:
                try:
                    print(tokens_to_fen(generated_fen))
                except:
                    pass


plt.subplot(121)
plt.plot(losses)
plt.xlabel("Steps")
plt.ylabel("Train loss")

plt.subplot(122)
plt.plot(log_probs)
plt.xlabel("Steps")
plt.ylabel("Validation log probs")

plt.show()
