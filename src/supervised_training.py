import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--continue_from_checkpoint", action="store_true")

    args = parser.parse_args()
    distributed = args.distributed
    continue_from_checkpoint = args.continue_from_checkpoint

    base_path = Path("./src")
    
    # ====================== LOAD CHECKPOINT ======================
    if continue_from_checkpoint: checkpoint = torch.load(base_path / "supervised_checkpoints" / "model_0200000.pt", map_location="cpu", weights_only=False)  # as the config was saved as well, cannot use weights_only=True

    # ====================== DEVICE ======================
    if distributed:
        assert torch.cuda.is_available()
        local_rank = int(os.environ["SLURM_PROCID"])
        rank = local_rank  # should be something different with multiple nodes
        world_size = int(os.environ["SLURM_NTASKS"])

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        master_process = rank == 0
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        master_process = True
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    device = torch.device(device)

    # ====================== CONFIG ======================
    # config = Config(n_layers=1, batch_size=1024, n_steps=50, train_logging_interval=1, validation_interval=5, save_interval=500, n_validation_generations=1, embed_dim=128, lr=1e-2, weight_decay=0)
    # config = Config(n_layers=1, batch_size=1024, n_steps=1000, train_logging_interval=1, validation_interval=100, save_interval=500, n_validation_generations=1)
    if continue_from_checkpoint:
        config = checkpoint["config"]
    else:
        config = Config(train_logging_interval=10, validation_interval=10000, n_steps=1_000_000, save_interval=20_000, batch_size=1024)

    # ====================== SEED AND PRECISION ======================
    torch.manual_seed(rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rank)
    torch.set_float32_matmul_precision("high")

    # ====================== LOGGING ======================
    if master_process:
        if continue_from_checkpoint:
            logging_path = base_path / "runs"/ "supervised" / "real_model_v1"
        else:
            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            logging_path = base_path / "runs"/ "supervised" / current_time

        train_writer = SummaryWriter(logging_path / "train")
        validation_writer = SummaryWriter(logging_path / "validation")

    # ====================== DATASET ======================
    dataset_path = base_path / "dataset"
    trainset = torch.load(dataset_path / "train" / "trainset.pt", weights_only=False, map_location="cpu")
    validationset = torch.load(dataset_path / "validation" / "validationset.pt", weights_only=False, map_location="cpu")
    # validationset = Subset(validationset, list(range(100)))  # do not use the entire validation set if using a cpu
    trainsampler = DistributedSampler(trainset, shuffle=True, rank=rank, num_replicas=world_size) if distributed else None
    validationsampler = DistributedSampler(validationset, shuffle=True, rank=rank, num_replicas=world_size) if distributed else None
    assert config.batch_size % world_size == 0
    local_batch_size = config.batch_size // world_size
    trainloader = DataLoader(trainset, batch_size=local_batch_size, shuffle=not distributed, sampler=trainsampler, num_workers=10 if distributed else 0, pin_memory=distributed)
    validationloader = DataLoader(validationset, batch_size=8 * local_batch_size, shuffle=not distributed, sampler=validationsampler, num_workers=10 if distributed else 0, pin_memory=distributed)

    # ====================== MODEL ======================
    model = MaskedDiffusion(config)
    if continue_from_checkpoint: model.load_state_dict(checkpoint["model"])
    model.to(device=device)

    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    model = torch.compile(model)

    # ====================== OPTIMIZER ======================
    params_adam = [p for p in model.parameters() if p.ndim != 2]
    params_muon = [p for p in model.parameters() if p.ndim == 2]
    adam = AdamW(params_adam, lr=config.lr, weight_decay=config.weight_decay)
    muon = Muon(params_muon, lr=config.lr, weight_decay=config.weight_decay)
    if continue_from_checkpoint: adam.load_state_dict(checkpoint["adam"])
    if continue_from_checkpoint: muon.load_state_dict(checkpoint["muon"])

    # ====================== LOSS FUNCTION ======================
    def compute_loss(model: MaskedDiffusion, fens, themes, ratings):
        # could use the variance reduced version in rl.espo, but for supervised learning, this is good enough (variance is not a problem)
        batch_size = len(ratings)
        t = (torch.rand(1) + torch.arange(1, batch_size + 1, 1) / batch_size) % 1
        alpha_t = config.masking_schedule(t).unsqueeze(1).to(device)
        random_mask = torch.rand(fens.size(), device=device) < alpha_t
        masked_fens = torch.where(random_mask, fens, config.mask_token)

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits = model(masked_fens, themes, ratings)
        unwrapped_model = model.module if distributed else model
        return unwrapped_model.elbo_loss(t, logits, fens, masked_fens)

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
        
        logits = logits / samples
        logits = F.log_softmax(logits, dim=1)

        all_indices = torch.arange(config.n_fen_tokens)
        if master_process: validation_writer.add_scalars("Impossible token logits/max",
                                        {"board": logits[:64, ~(all_indices <= FENTokens.black_king)].max(),
                                        "side": logits[64:65, ~((all_indices >= FENTokens.side_white) & (all_indices <= FENTokens.side_black))].max(),
                                        "castle": logits[65:69, ~((all_indices >= FENTokens.no_castle) & (all_indices <= FENTokens.castle_black_queen))].max(),
                                        "enpassant": logits[69:71, ~((all_indices >= FENTokens.no_en_passant) & (all_indices <= FENTokens.en_passant_h))].max(),
                                        "halfmove": logits[71:73, ~((all_indices >= FENTokens.pad_counter) & (all_indices <= FENTokens.counter_9))].max(),
                                        "fullmove": logits[73:76, ~((all_indices >= FENTokens.pad_counter) & (all_indices <= FENTokens.counter_9))].max()}, step)
        if master_process: validation_writer.add_scalars("Impossible token logits/mean",
                                        {"board": logits[:64, ~(all_indices <= FENTokens.black_king)].mean(),
                                        "side": logits[64:65, ~((all_indices >= FENTokens.side_white) & (all_indices <= FENTokens.side_black))].mean(),
                                        "castle": logits[65:69, ~((all_indices >= FENTokens.no_castle) & (all_indices <= FENTokens.castle_black_queen))].mean(),
                                        "enpassant": logits[69:71, ~((all_indices >= FENTokens.no_en_passant) & (all_indices <= FENTokens.en_passant_h))].mean(),
                                        "halfmove": logits[71:73, ~((all_indices >= FENTokens.pad_counter) & (all_indices <= FENTokens.counter_9))].mean(),
                                        "fullmove": logits[73:76, ~((all_indices >= FENTokens.pad_counter) & (all_indices <= FENTokens.counter_9))].mean()}, step)
        if master_process: validation_writer.add_scalars("Possible logits/mean",
                                        {"board": logits[:64, (all_indices <= FENTokens.black_king)].mean(),
                                        "side": logits[64:65, ((all_indices >= FENTokens.side_white) & (all_indices <= FENTokens.side_black))].mean(),
                                        "castle": logits[65:69, ((all_indices >= FENTokens.no_castle) & (all_indices <= FENTokens.castle_black_queen))].mean(),
                                        "enpassant": logits[69:71, ((all_indices >= FENTokens.no_en_passant) & (all_indices <= FENTokens.en_passant_h))].mean(),
                                        "halfmove": logits[71:73, ((all_indices >= FENTokens.pad_counter) & (all_indices <= FENTokens.counter_9))].mean(),
                                        "fullmove": logits[73:76, ((all_indices >= FENTokens.pad_counter) & (all_indices <= FENTokens.counter_9))].mean()}, step)
        if master_process: validation_writer.add_image("Probabilities", logits.unsqueeze(0).exp(), step)

    def write_fen(step):
        validation_themes = torch.zeros((config.n_validation_generations, config.n_themes), dtype=torch.float32, device=device)
        indices = torch.randint(0, config.n_themes, (config.n_validation_generations,))
        validation_themes[:, indices] = 1
        validation_ratings = scale_ratings(3000 * torch.rand(config.n_validation_generations, dtype=torch.float32, device=device) + 300)
        unwrapped_model = model.module if distributed else model
        fens = unwrapped_model.sample(validation_themes, validation_ratings, steps=128)
        for generated_fen in fens:
            try:
                string = tokens_to_fen(generated_fen)
                if master_process: validation_writer.add_text("Generations/fen", string, step)
            except:
                pass

    def compute_validation_loss():
        total = torch.zeros(2, dtype=torch.float32, device=device)
        with torch.no_grad():
            for validation_fen, validation_theme, validation_rating in validationloader:
                validation_fen, validation_theme, validation_rating = validation_fen.to(dtype=torch.long, device=device), validation_theme.to(dtype=torch.float32, device=device), validation_rating.to(dtype=torch.float32, device=device)
                validation_loss = compute_loss(model, validation_fen, validation_theme, validation_rating).sum()
                total[0] += validation_loss.detach()
                total[1] += len(validation_rating)
        if distributed:
            all_reduce(total, op=ReduceOp.SUM)
        return total[0] / total[1]

    def write_validation_loss(step):
        validation_loss = compute_validation_loss()
        if master_process:
            validation_writer.add_scalar("Loss", validation_loss, step)

    def save_state():
        checkpoint_path = base_path / "supervised_checkpoints" / f"model_{step:07d}.pt"
        base_model = model.module if distributed else model
        if hasattr(base_model, "_orig_mod"):
            base_model = base_model._orig_mod
        checkpoint = {
            "model": base_model.state_dict(),
            "config": config,
            "adam": adam.state_dict(),
            "muon": muon.state_dict(),
            "epoch": epoch,
            "step": step
        }
        torch.save(checkpoint, checkpoint_path)

    # ====================== TRAINING LOOP ======================
    step = 0
    epoch = 0
    if continue_from_checkpoint: step = checkpoint["step"]
    if continue_from_checkpoint: epoch = checkpoint["epoch"]
    ended = False
    total = torch.zeros(3, dtype=torch.float32, device=device)
    while not ended:
        epoch += 1
        if distributed:
            trainsampler.set_epoch(epoch)

        for fen, theme, rating in trainloader:
            fen, theme, rating = fen.to(dtype=torch.long, device=device), theme.to(dtype=torch.float32, device=device), rating.to(dtype=torch.float32, device=device)

            adam.zero_grad()
            muon.zero_grad()

            loss = compute_loss(model, fen, theme, rating)
            loss.mean().backward()

            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            adam.step()
            muon.step()

            step += 1

            total[0] += loss.detach().sum()
            total[1] += len(rating) * norm.detach()
            total[2] += len(rating)
            if step % config.train_logging_interval == 0 or step == 1:
                if distributed: all_reduce(total, op=ReduceOp.SUM)
                if master_process: train_writer.add_scalar("Loss", total[0] / total[2], step)
                if master_process: train_writer.add_scalar("Grad norm", total[1] / total[2], step)
                total = torch.zeros(3, dtype=torch.float32, device=device)

            # validation
            if step % config.validation_interval == 0 or step == 1:
                model.eval()
                if distributed:
                    validationsampler.set_epoch(step // config.validation_interval)

                write_logits(step)
                write_fen(step)
                write_validation_loss(step)

                model.train()
            
            # create a checkpoint
            if step % config.save_interval == 0 and master_process:
                save_state()

            # are we finished?
            if step >= config.n_steps:
                ended = True
                break

    model.eval()
    validation_loss = compute_validation_loss()
    if master_process:
        train_writer.add_hparams({
            "masking_schedule": config.schedule,
            "n_heads": config.n_heads,
            "n_layers": config.n_layers,
            "embed_dim": config.embed_dim,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "batch_size": config.batch_size,
            "n_steps": config.n_steps,
            "validation_interval": config.validation_interval,
            "train_logging_interval": config.train_logging_interval,
            "save_interval": config.save_interval,
            "n_validation_generations": config.n_validation_generations,
        }, {
            "validation_loss": validation_loss
        })

    if master_process: train_writer.close()
    if master_process: validation_writer.close()
    if distributed: destroy_process_group()


if __name__ == "__main__":
    main()
