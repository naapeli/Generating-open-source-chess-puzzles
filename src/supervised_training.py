import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, Muon
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from datetime import datetime

from MaskedDiffusion.model import MaskedDiffusion
from Config import Config
from tokenization.tokenization import tokens_to_fen, scale_ratings, FENTokens


base_path = Path("./src")
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
logging_path = base_path / "runs"/ "supervised" / current_time
train_writer = SummaryWriter(logging_path / "train")
validation_writer = SummaryWriter(logging_path / "validation")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
config = Config(n_layers=1, batch_size=32, n_steps=1000, validation_interval=25, n_validation_generations=1, embed_dim=128)

dataset_path = base_path / "dataset"
trainset = torch.load(dataset_path / "train" / "trainset.pt", weights_only=False)
validationset = torch.load(dataset_path / "validation" / "validationset.pt", weights_only=False)
trainloader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True)
validationloader = DataLoader(validationset, batch_size=config.batch_size, shuffle=True)

model = MaskedDiffusion(config)
model.to(device=device, dtype=torch.float32)

params_adam = [p for p in model.parameters() if p.ndim != 2]
params_muon = [p for p in model.parameters() if p.ndim == 2]
adam = AdamW(params_adam, lr=config.lr, weight_decay=config.weight_decay)
muon = Muon(params_muon, lr=config.lr, weight_decay=config.weight_decay)

def compute_loss(model: MaskedDiffusion, fens, themes, ratings):
    batch_size = len(ratings)
    t = (torch.rand(1) + torch.arange(1, batch_size + 1, 1) / batch_size) % 1
    alpha_t = model.config.masking_schedule(t).unsqueeze(1)
    random_mask = torch.rand(fens.size()) < alpha_t
    masked_fens = torch.where(random_mask, fens, model.config.mask_token)

    logits = model(masked_fens, themes, ratings)
    return model.elbo_loss(t, logits, fens, masked_fens)

step = 0
ended = False
while not ended:
    for fen, theme, rating in trainloader:
        fen, theme, rating = fen.to(dtype=torch.long, device=device), theme.to(dtype=torch.float32, device=device), rating.to(dtype=torch.float32, device=device)

        adam.zero_grad()
        muon.zero_grad()

        loss = compute_loss(model, fen, theme, rating)
        
        loss.backward()
        adam.step()
        muon.step()

        step += 1

        # print(f"Steps: {step} - Train loss: {loss.item()}")
        train_writer.add_scalar("Loss", loss, step)

        if step % config.validation_interval == 0:
            losses = []
            logits = []

            for validation_fen, validation_theme, validation_rating in validationloader:
                validation_fen, validation_theme, validation_rating = validation_fen.to(dtype=torch.long, device=device), validation_theme.to(dtype=torch.float32, device=device), validation_rating.to(dtype=torch.float32, device=device)
                with torch.no_grad():
                    validation_loss = compute_loss(model, validation_fen, validation_theme, validation_rating)
                losses.append(validation_loss.item())

                with torch.no_grad():
                    logits.append(model(validation_fen, validation_theme, validation_rating))
            
            validation_writer.add_scalar("Loss", sum(losses) / len(losses), step)

            logits = torch.sum(sum(logits), dim=0) / (config.batch_size * len(logits))
            all_indices = torch.arange(logits.size(-1))
            validation_writer.add_scalars("Impossible token logits/max", {"board": logits[:64, ~(all_indices <= FENTokens.black_king)].max(),
                                          "side": logits[64:65, ~((all_indices >= FENTokens.side_white) & (all_indices <= FENTokens.side_black))].max(),
                                          "castle": logits[65:69, ~((all_indices >= FENTokens.no_castle) & (all_indices <= FENTokens.castle_black_queen))].max(),
                                          "enpassant": logits[69:71, ~((all_indices >= FENTokens.no_en_passant) & (all_indices <= FENTokens.en_passant_h))].max(),
                                          "halfmove": logits[71:73, ~((all_indices >= FENTokens.pad_counter) & (all_indices <= FENTokens.counter_9))].max(),
                                          "fullmove": logits[73:76, ~((all_indices >= FENTokens.pad_counter) & (all_indices <= FENTokens.counter_9))].max()}, step)
            validation_writer.add_scalars("Impossible token logits/mean", {"board": logits[:64, ~(all_indices <= FENTokens.black_king)].mean(),
                                          "side": logits[64:65, ~((all_indices >= FENTokens.side_white) & (all_indices <= FENTokens.side_black))].mean(),
                                          "castle": logits[65:69, ~((all_indices >= FENTokens.no_castle) & (all_indices <= FENTokens.castle_black_queen))].mean(),
                                          "enpassant": logits[69:71, ~((all_indices >= FENTokens.no_en_passant) & (all_indices <= FENTokens.en_passant_h))].mean(),
                                          "halfmove": logits[71:73, ~((all_indices >= FENTokens.pad_counter) & (all_indices <= FENTokens.counter_9))].mean(),
                                          "fullmove": logits[73:76, ~((all_indices >= FENTokens.pad_counter) & (all_indices <= FENTokens.counter_9))].mean()}, step)
            validation_writer.add_scalars("Possible logits/mean", {"board": logits[:64, (all_indices <= FENTokens.black_king)].mean(),
                                          "side": logits[64:65, ((all_indices >= FENTokens.side_white) & (all_indices <= FENTokens.side_black))].mean(),
                                          "castle": logits[65:69, ((all_indices >= FENTokens.no_castle) & (all_indices <= FENTokens.castle_black_queen))].mean(),
                                          "enpassant": logits[69:71, ((all_indices >= FENTokens.no_en_passant) & (all_indices <= FENTokens.en_passant_h))].mean(),
                                          "halfmove": logits[71:73, ((all_indices >= FENTokens.pad_counter) & (all_indices <= FENTokens.counter_9))].mean(),
                                          "fullmove": logits[73:76, ((all_indices >= FENTokens.pad_counter) & (all_indices <= FENTokens.counter_9))].mean()}, step)
            validation_writer.add_image("Logits", logits.unsqueeze(0) / 10 + 0.5, step)

            validation_themes = torch.zeros((config.n_validation_generations, config.n_themes), dtype=torch.float32)
            indices = torch.randint(0, config.n_themes, (config.n_validation_generations,))
            validation_themes[:, indices] = 1
            validation_ratings = scale_ratings(3000 * torch.rand(config.n_validation_generations, dtype=torch.float32) + 300)
            fens = model.sample(validation_themes, validation_ratings, steps=128)
            # print(fens.squeeze().numpy())
            for generated_fen in fens:
                try:
                    string = tokens_to_fen(generated_fen)
                    validation_writer.add_text("Generations/fen", string, step)
                except:
                    pass
        
        if step >= config.n_steps:
            ended = True
            break

train_writer.close()
validation_writer.close()
