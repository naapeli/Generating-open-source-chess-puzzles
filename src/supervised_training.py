import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW, Muon
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
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
from tokenization.tokenization import tokens_to_fen, tokens_to_move, scale_ratings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--checkpoint_name", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()
    distributed = args.distributed
    continue_from_checkpoint = args.checkpoint_name != None

    base_path = Path("./src")
    
    run_name = args.run_name
    if not run_name:
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    logging_path = base_path / "runs" / "supervised" / run_name

    # ====================== LOAD CHECKPOINT ======================
    if continue_from_checkpoint:
        checkpoint = torch.load(logging_path / args.checkpoint_name, map_location="cpu", weights_only=False)

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
    if continue_from_checkpoint:
        config = checkpoint["config"]
    else:
        config = Config(train_logging_interval=10, validation_interval=10_000, n_steps=1_000_000, save_interval=100_000, batch_size=1024)
        
    # ====================== SEED AND PRECISION ======================
    torch.manual_seed(rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rank)
    torch.set_float32_matmul_precision("high")

    # ====================== LOGGING ======================
    if master_process:
        logging_path.mkdir(parents=True, exist_ok=True)
        train_writer = SummaryWriter(logging_path / "train")
        validation_writer = SummaryWriter(logging_path / "validation")

        if not continue_from_checkpoint:
            import yaml
            hparams = {
                "predict_moves": config.predict_moves,
                "masking_schedule": config.schedule,
                "n_heads": config.n_heads,
                "n_layers": config.n_layers,
                "embed_dim": config.embed_dim,
                "lr": config.lr,
                "weight_decay": config.weight_decay,
                "batch_size": config.batch_size,
                "n_steps": config.n_steps,
                "use_context": config.use_context,
            }
            with open(logging_path / "config.yml", "w") as f:
                yaml.dump(hparams, f)

    # ====================== DATASET ======================
    dataset_path = base_path / "dataset"
    trainset = torch.load(dataset_path / "with_best_move" / "trainset.pt", weights_only=False, map_location="cpu")
    testset = torch.load(dataset_path / "with_best_move" / "testset.pt", weights_only=False, map_location="cpu")
    trainsampler = DistributedSampler(trainset, shuffle=True, rank=rank, num_replicas=world_size) if distributed else None
    validationsampler = DistributedSampler(testset, shuffle=True, rank=rank, num_replicas=world_size) if distributed else None
    assert config.batch_size % world_size == 0
    local_batch_size = config.batch_size // world_size
    trainloader = DataLoader(trainset, batch_size=local_batch_size, shuffle=not distributed, sampler=trainsampler, num_workers=10 if distributed else 0, pin_memory=distributed)
    validationloader = DataLoader(testset, batch_size=8 * local_batch_size, shuffle=not distributed, sampler=validationsampler, num_workers=10 if distributed else 0, pin_memory=distributed)

    # ====================== MODEL ======================
    model = MaskedDiffusion(config)
    if continue_from_checkpoint: model.load_state_dict(checkpoint["model"])
    model.to(device=device)

    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    model = torch.compile(model)

    # ====================== OPTIMIZER ======================
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler1 = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=1000)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=config.n_steps, eta_min=0.1 * config.lr)
    lr_scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[1000], last_epoch=checkpoint["step"] if continue_from_checkpoint else -1)
    if continue_from_checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception as e:
            pass

    # ====================== LOSS FUNCTION ======================
    def compute_loss(model: MaskedDiffusion, fens, moves, themes=None, ratings=None):
        # could use the variance reduced version in rl.espo, but for supervised learning, this is good enough (variance is not a problem)
        tokens = torch.cat([fens, moves], dim=1) if config.predict_moves else fens
        batch_size = len(fens)
        t = (torch.rand(1) + torch.arange(1, batch_size + 1, 1) / batch_size) % 1
        if master_process:
            train_writer.add_scalars("t", {"min": t.min(), "max": t.max(), "mean": t.mean()}, step)
            train_writer.add_scalars("alpha_t", {"min": config.masking_schedule(t).min(), "max": config.masking_schedule(t).max(), "mean": config.masking_schedule(t).mean()}, step)
            train_writer.add_scalars("t_weight", {"min": config.masking_schedule.get_weight(t).min(), "max": config.masking_schedule.get_weight(t).max(), "mean": config.masking_schedule.get_weight(t).mean()}, step)
        alpha_t = config.masking_schedule(t).unsqueeze(1).to(device)

        random_mask = torch.rand(tokens.size(), device=device) < alpha_t
        masked_tokens = torch.where(random_mask, tokens, config.mask_token)

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            if config.use_context:
                logits = model(masked_tokens, themes, ratings)
            else:
                logits = model(masked_tokens)
        unwrapped_model = model.module if distributed else model
        return unwrapped_model.elbo_loss(t, logits, tokens, masked_tokens)

    # ====================== VALIDATION ======================
    def write_logits(step):
        seq_len = config.fen_length + config.move_length if config.predict_moves else config.fen_length
        probs_sum = torch.zeros((seq_len, config.n_tokens), dtype=torch.float32, device=device)
        samples = torch.tensor(0, device=device)
        for validation_fen, validation_move, validation_theme, validation_rating in validationloader:
            validation_fen, validation_move, validation_theme, validation_rating = validation_fen.to(dtype=torch.long, device=device), validation_move.to(dtype=torch.long, device=device), validation_theme.to(dtype=torch.float32, device=device), validation_rating.to(dtype=torch.float32, device=device)
            tokens = torch.cat([validation_fen, validation_move], dim=1) if config.predict_moves else validation_fen
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    if config.use_context:
                        sub_logits = model(tokens, validation_theme, validation_rating)
                    else:
                        sub_logits = model(tokens)
                sub_probs = F.softmax(sub_logits, dim=2).to(dtype=torch.float32)
                probs_sum += sub_probs.sum(dim=0)
                samples += len(validation_fen)
        if distributed:
            all_reduce(probs_sum, op=ReduceOp.SUM)
            all_reduce(samples, op=ReduceOp.SUM)
        
        avg_probs = probs_sum / samples

        if master_process: validation_writer.add_image("Probabilities", avg_probs.unsqueeze(0), step)

    def write_fen(step):
        unwrapped_model = model.module if distributed else model
        if config.use_context:
            validation_themes = torch.zeros((config.n_validation_generations, config.n_themes), dtype=torch.float32, device=device)
            indices = torch.randint(0, config.n_themes, (config.n_validation_generations,))
            validation_themes[:, indices] = 1
            validation_ratings = scale_ratings(3000 * torch.rand(config.n_validation_generations, dtype=torch.float32, device=device) + 300)
            tokens = unwrapped_model.sample(validation_themes, validation_ratings, steps=512)
        else:
            tokens = unwrapped_model.sample(batch_size=config.n_validation_generations, steps=512)
        for generated_tokens in tokens:
            try:
                fen_str = tokens_to_fen(generated_tokens[:config.fen_length])
                if config.predict_moves:
                    move_str = tokens_to_move(generated_tokens[config.fen_length:])
                    if master_process: validation_writer.add_text("Generations/fen", f"{fen_str} move: {move_str}", step)
                else:
                    if master_process: validation_writer.add_text("Generations/fen", fen_str, step)
            except:
                pass

    def compute_validation_loss():
        total = torch.zeros(2, dtype=torch.float32, device=device)
        with torch.no_grad():
            for validation_fen, validation_move, validation_theme, validation_rating in validationloader:
                validation_fen, validation_move, validation_theme, validation_rating = validation_fen.to(dtype=torch.long, device=device), validation_move.to(dtype=torch.long, device=device), validation_theme.to(dtype=torch.float32, device=device), validation_rating.to(dtype=torch.float32, device=device)
                if config.use_context:
                    validation_loss = compute_loss(model, validation_fen, validation_move, validation_theme, validation_rating).sum()
                else:
                    validation_loss = compute_loss(model, validation_fen, validation_move, None, None).sum()
                total[0] += validation_loss.detach()
                total[1] += len(validation_fen)
        if distributed:
            all_reduce(total, op=ReduceOp.SUM)
        return total[0] / total[1]

    def write_validation_loss(step):
        validation_loss = compute_validation_loss()
        if master_process:
            validation_writer.add_scalar("Loss", validation_loss, step)

    def save_state():
        checkpoint_path = logging_path / f"model_{step:07d}.pt"
        base_model = model.module if distributed else model
        if hasattr(base_model, "_orig_mod"):
            base_model = base_model._orig_mod
        checkpoint = {
            "model": base_model.state_dict(),
            "config": config,
            "optimizer": optimizer.state_dict(),
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

        for fen, move, theme, rating in trainloader:
            fen, move, theme, rating = fen.to(dtype=torch.long, device=device), move.to(dtype=torch.long, device=device), theme.to(dtype=torch.float32, device=device), rating.to(dtype=torch.float32, device=device)

            optimizer.zero_grad()

            if config.use_context:
                loss = compute_loss(model, fen, move, theme, rating)
            else:
                loss = compute_loss(model, fen, move, None, None)
            loss.mean().backward()

            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

            step += 1

            total[0] += loss.detach().sum()
            total[1] += len(fen) * norm.detach()
            total[2] += len(fen)
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
            "use_context": config.use_context,
        }, {
            "validation_loss": validation_loss
        })

    if master_process: train_writer.close()
    if master_process: validation_writer.close()
    if distributed: destroy_process_group()


if __name__ == "__main__":
    main()
