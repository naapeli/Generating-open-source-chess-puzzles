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

from RatingModel.model import RatingModel
from Config import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--continue_from_checkpoint", action="store_true")

    args = parser.parse_args()
    distributed = args.distributed
    continue_from_checkpoint = args.continue_from_checkpoint

    base_path = Path("./src")
    
    # ====================== LOAD CHECKPOINT ======================
    if continue_from_checkpoint: checkpoint = torch.load(base_path / "rating_model_checkpoints" / "model_0004500.pt", map_location="cpu", weights_only=False)  # as the config was saved as well, cannot use weights_only=True

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
    device = torch.device(device)

    # ====================== CONFIG ======================
    if continue_from_checkpoint:
        config = checkpoint["config"]
        config.n_steps = 20_000
    else:
        config = Config(train_logging_interval=1, validation_interval=50, n_steps=5000, save_interval=500, batch_size=512)

    # ====================== SEED AND PRECISION ======================
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    torch.set_float32_matmul_precision("high")

    # ====================== LOGGING ======================
    if master_process:
        if continue_from_checkpoint:
            logging_path = base_path / "runs"/ "rating_model" / "v1"
        else:
            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            logging_path = base_path / "runs"/ "rating_model" / current_time

        train_writer = SummaryWriter(logging_path / "train")
        validation_writer = SummaryWriter(logging_path / "validation")

    # ====================== DATASET ======================
    dataset_path = base_path / "dataset"
    trainset = torch.load(dataset_path / "train" / "trainset.pt", weights_only=False, map_location="cpu")
    validationset = torch.load(dataset_path / "validation" / "validationset.pt", weights_only=False, map_location="cpu")
    trainsampler = DistributedSampler(trainset, shuffle=True, rank=rank, num_replicas=world_size) if distributed else None
    validationsampler = DistributedSampler(validationset, shuffle=True, rank=rank, num_replicas=world_size) if distributed else None
    assert config.batch_size % world_size == 0
    local_batch_size = config.batch_size // world_size
    trainloader = DataLoader(trainset, batch_size=local_batch_size, shuffle=not distributed, sampler=trainsampler, num_workers=10 if distributed else 0, pin_memory=distributed)
    validationloader = DataLoader(validationset, batch_size=8 * local_batch_size, shuffle=not distributed, sampler=validationsampler, num_workers=10 if distributed else 0, pin_memory=distributed)

    # ====================== MODEL ======================
    model = RatingModel(config)
    if continue_from_checkpoint: model.load_state_dict(checkpoint["model"])
    model.to(device=device)

    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ====================== OPTIMIZER ======================
    params_adam = [p for p in model.parameters() if p.ndim != 2]
    params_muon = [p for p in model.parameters() if p.ndim == 2]
    adam = AdamW(params_adam, lr=config.lr, weight_decay=config.weight_decay)
    muon = Muon(params_muon, lr=config.lr, weight_decay=config.weight_decay)
    if continue_from_checkpoint: adam.load_state_dict(checkpoint["adam"])
    if continue_from_checkpoint: muon.load_state_dict(checkpoint["muon"])

    # ====================== VALIDATION ======================
    def compute_validation_loss():
        total = torch.zeros(2, dtype=torch.float32, device=device)
        with torch.no_grad():
            for validation_fen, validation_theme, validation_rating in validationloader:
                validation_fen, validation_theme, validation_rating = validation_fen.to(dtype=torch.long, device=device), validation_theme.to(dtype=torch.float32, device=device), validation_rating.to(dtype=torch.float32, device=device)
                predicted_ratings = model(validation_fen, validation_theme)
                validation_loss = F.mse_loss(predicted_ratings, validation_rating, reduction="sum")
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
        checkpoint_path = base_path / "rating_model_checkpoints" / f"model_{step:07d}.pt"
        checkpoint = {
            "model": model.module.state_dict() if distributed else model.state_dict(),
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
    while not ended:
        epoch += 1
        if distributed:
            trainsampler.set_epoch(epoch)

        total = torch.zeros(3, dtype=torch.float32, device=device)
        for fen, theme, rating in trainloader:
            fen, theme, rating = fen.to(dtype=torch.long, device=device), theme.to(dtype=torch.float32, device=device), rating.to(dtype=torch.float32, device=device)

            adam.zero_grad()
            muon.zero_grad()

            predicted_ratings = model(fen, theme)
            loss = F.mse_loss(predicted_ratings, rating, reduction="none")
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
                if distributed:
                    validationsampler.set_epoch(step // config.validation_interval)

                write_validation_loss(step)
            
            # create a checkpoint
            if step % config.save_interval == 0 and master_process:
                save_state()

            # are we finished?
            if step >= config.n_steps:
                ended = True
                break

    validation_loss = compute_validation_loss()
    if master_process:
        train_writer.add_hparams({
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
