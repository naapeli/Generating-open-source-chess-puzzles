import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW, Muon
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp

from pathlib import Path
from datetime import datetime
import os
import argparse

from MaskedDiffusion.model import MaskedDiffusion
from Config import Config
from tokenization.tokenization import tokens_to_fen, scale_ratings, FENTokens


parser = argparse.ArgumentParser()
parser.add_argument("--distributed")

args = parser.parse_args()
distributed = args.distributed is not None

# ====================== DEVICE ======================
if distributed:
    assert torch.cuda.is_available()
    local_rank = int(os.environ["SLURM_PROCID"])
    rank = local_rank  # should be something different with multiple nodes
    world_size = int(os.environ["SLURM_NTASKS"])
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)
    master_process = rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if device.startswith("cuda") else "cpu"
device = torch.device(device)

# ====================== CONFIG ======================
# config = Config(n_layers=1, batch_size=1024, n_steps=1000, train_logging_interval=10, validation_interval=25, save_interval=100, n_validation_generations=1, embed_dim=128)
config = Config(n_layers=1, batch_size=1024, n_steps=1000, train_logging_interval=10000, validation_interval=25000, save_interval=100000, n_validation_generations=1, embed_dim=128, lr=1e-2, weight_decay=0)

# ====================== SEED AND PRECISION ======================
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
torch.set_float32_matmul_precision("high")

# ====================== LOGGING ======================
base_path = Path("./src")
if master_process:
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    logging_path = base_path / "runs"/ "supervised" / current_time
    train_writer = SummaryWriter(logging_path / "train")
    validation_writer = SummaryWriter(logging_path / "validation")

# ====================== DATASET ======================
dataset_path = base_path / "dataset"
trainset = torch.load(dataset_path / "train" / "trainset.pt", weights_only=False, map_location="cpu")

fen, theme, rating = trainset[0]
fen, theme, rating = fen.unsqueeze(0), theme.unsqueeze(0), rating.unsqueeze(0)

validationset = torch.load(dataset_path / "validation" / "validationset.pt", weights_only=False, map_location="cpu")
trainsampler = DistributedSampler(trainset, shuffle=True, rank=ddp_rank, num_replicas=ddp_world_size) if distributed else None
validationsampler = DistributedSampler(validationset, shuffle=True, rank=ddp_rank, num_replicas=ddp_world_size) if distributed else None
assert config.batch_size % ddp_world_size == 0
local_batch_size = config.batch_size // ddp_world_size
trainloader = DataLoader(trainset, batch_size=local_batch_size, shuffle=not distributed, sampler=trainsampler, num_workers=os.cpu_count() // ddp_world_size, pin_memory=True)
validationloader = DataLoader(validationset, batch_size=local_batch_size, shuffle=not distributed, sampler=validationsampler, num_workers=os.cpu_count() // ddp_world_size, pin_memory=True)

# ====================== MODEL ======================
model = MaskedDiffusion(config)
model.to(device=device)

if distributed:
    model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

# ====================== OPTIMIZER ======================
params_adam = [p for p in model.parameters() if p.ndim != 2]
params_muon = [p for p in model.parameters() if p.ndim == 2]
adam = AdamW(params_adam, lr=config.lr, weight_decay=config.weight_decay)
# muon = Muon(params_muon, lr=config.lr, weight_decay=config.weight_decay)
muon = AdamW(params_muon, lr=config.lr, weight_decay=config.weight_decay)

# ====================== LOSS FUNCTION ======================
def compute_loss(model: MaskedDiffusion, fens, themes, ratings):
    batch_size = len(ratings)
    t = (torch.rand(1) + torch.arange(1, batch_size + 1, 1) / batch_size) % 1
    alpha_t = model.config.masking_schedule(t).unsqueeze(1).to(device)
    random_mask = torch.rand(fens.size(), device=device) < alpha_t
    masked_fens = torch.where(random_mask, fens, model.config.mask_token)

    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        logits = model(masked_fens, themes, ratings)
    return model.elbo_loss(t, logits, fens, masked_fens)

# ====================== VALIDATION ======================
def write_logits(step):
    logits = torch.zeros((config.fen_length, config.n_fen_tokens), dtype=torch.float32, device=device)
    samples = torch.tensor(0, device=device)
    for validation_fen, validation_theme, validation_rating in validationloader:
        validation_fen, validation_theme, validation_rating = validation_fen.to(dtype=torch.long, device=device), validation_theme.to(dtype=torch.float32, device=device), validation_rating.to(dtype=torch.float32, device=device)
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                sub_logits = model(validation_fen, validation_theme, validation_rating)
            logits += sub_logits.sum(dim=0)
            samples += len(validation_rating)
    if distributed:
        all_reduce(logits, op=ReduceOp.SUM)
        all_reduce(samples, op=ReduceOp.SUM)
    if master_process:
        logits = logits / samples
        logits = F.log_softmax(logits, dim=1)  # plot the logits with respect to the other logits

        all_indices = torch.arange(config.n_fen_tokens)
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

def write_fen(step):
    if master_process:
        validation_themes = torch.zeros((config.n_validation_generations, config.n_themes), dtype=torch.float32, device=device)
        indices = torch.randint(0, config.n_themes, (config.n_validation_generations,))
        validation_themes[:, indices] = 1
        validation_ratings = scale_ratings(3000 * torch.rand(config.n_validation_generations, dtype=torch.float32, device=device) + 300)
        unwrapped_model = model.module if distributed else model
        fens = unwrapped_model.sample(validation_themes, validation_ratings, steps=128)
        for generated_fen in fens:
            try:
                string = tokens_to_fen(generated_fen)
                validation_writer.add_text("Generations/fen", string, step)
            except:
                pass

def write_validation_loss(step):
    total = torch.zeros(2, dtype=torch.float32, device=device)
    with torch.no_grad():
        for validation_fen, validation_theme, validation_rating in validationloader:
            validation_fen, validation_theme, validation_rating = validation_fen.to(dtype=torch.long, device=device), validation_theme.to(dtype=torch.float32, device=device), validation_rating.to(dtype=torch.float32, device=device)
            validation_loss = compute_loss(model, validation_fen, validation_theme, validation_rating).sum()
            total[0] += validation_loss.detach()
            total[1] += len(validation_rating)
    if distributed:
        all_reduce(total, op=ReduceOp.SUM)
    if master_process:
        validation_writer.add_scalar("Loss", total[0] / total[1], step)

def save_state():
    checkpoint_path = base_path / "supervised_checkpoints" / f"model_{step:05d}.pt"
    checkpoint = {
        "model": model.module.state_dict() if distributed else model.state_dict(),
        "config": config if distributed else config,
        "adam": adam.state_dict(),
        "muon": muon.state_dict(),
        "epoch": epoch,
        "step": step
    }
    torch.save(checkpoint, checkpoint_path)

# ====================== TRAINING LOOP ======================
step = 0
epoch = 0
ended = False
while not ended:
    epoch += 1
    if distributed:
        trainsampler.set_epoch(epoch)

    total = torch.zeros(3, dtype=torch.float32, device=device)
    # for fen, theme, rating in trainloader:
    for _ in range(1):
        fen, theme, rating = fen.to(dtype=torch.long, device=device), theme.to(dtype=torch.float32, device=device), rating.to(dtype=torch.float32, device=device)

        adam.zero_grad()
        muon.zero_grad()

        loss = compute_loss(model, fen, theme, rating).mean()
        loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        adam.step()
        muon.step()

        step += 1


        print(f"Step: {step} - loss: {loss.item():.2f}")
        fens = model.sample(theme, rating, steps=128)
        print(fens)
        for generated_fen in fens:
            try:
                string = tokens_to_fen(generated_fen)
                print(string)
            except:
                pass






        total[0] += len(rating) * loss.detach()
        total[1] += len(rating) * norm.detach()
        total[2] += len(rating)
        if step % config.train_logging_interval == 0:
            if distributed: all_reduce(total, op=ReduceOp.SUM)
            if master_process: train_writer.add_scalar("Loss", total[0] / total[2], step)
            if master_process: train_writer.add_scalar("Grad norm", total[1] / total[2], step)
            total = torch.zeros(3, dtype=torch.float32, device=device)

        # validation
        if step % config.validation_interval == 0:
            if distributed:
                validationsampler.set_epoch(step // config.validation_interval)

            write_logits(step)
            write_fen(step)
            write_validation_loss(step)
        
        # create a checkpoint
        if step % config.save_interval == 0 and master_process:
            save_state()

            ended = True
            break

        # are we finished?
        if step >= config.n_steps:
            ended = True
            break

if master_process: train_writer.close()
if master_process: validation_writer.close()
if distributed: destroy_process_group()
