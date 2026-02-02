import torch
import numpy as np

import random


n_quadrature = 10
t_points, quadrature_weights = np.polynomial.legendre.leggauss(n_quadrature)  # estimate the integral with a gaussian quadrature (https://arxiv.org/pdf/2510.08554)
t_points, quadrature_weights = torch.from_numpy(t_points), torch.from_numpy(quadrature_weights)
t_points, quadrature_weights = (1 - 0) / 2 * t_points + (1 + 0) / 2, (1 - 0) / 2 * quadrature_weights  # https://en.wikipedia.org/wiki/Gaussian_quadrature#Change_of_interval


def kl_estimate(model_elbo, reference_elbo):  # k2 from http://joschu.net/blog/kl-approx.html and https://arxiv.org/pdf/2512.03759 (biased, but low variance estimator)
    return 0.5 * torch.mean((model_elbo - reference_elbo) ** 2)

def espo_loss(model, reference_model, fens, themes, ratings, rewards, group_size, eps=0.2, beta=0.1):
    n_samples, sequence_length = fens.shape  # fens.shape == (batch_size * group_size, sequence_length)  batch_size groups of size group_size
    assert n_samples % group_size == 0
    batch_size = n_samples // group_size

    device = ratings.device
    config = model.config

    # t = t_points[None, :].expand(batch_size * group_size, n_quadrature).reshape(-1).to(device)  # batch_size * group_size * n_quadrature
    t = t_points.repeat(batch_size * group_size).to(device)  # batch_size * group_size * n_quadrature
    # t = ((torch.rand(1) + torch.arange(n_samples) / n_samples) % 1).to(device)
    alpha_t = config.masking_schedule(t)
    quadrature_weight = quadrature_weights.unsqueeze(0).to(device)

    # (batch_size * group_size * n_quadrature, ...)
    fens = fens.repeat_interleave(n_quadrature, dim=0)
    themes = themes.repeat_interleave(n_quadrature, dim=0)
    ratings = ratings.repeat_interleave(n_quadrature, dim=0)

    random_mask = torch.rand(fens.size(), device=device) < alpha_t.unsqueeze(1)
    masked_fens = torch.where(random_mask, fens, config.mask_token)

    # use the same mask for both models (antithetic sampling from https://arxiv.org/pdf/2512.03759)
    logits = model(masked_fens, themes, ratings)
    model_loss = model.elbo_loss(t, logits, fens, masked_fens)
    with torch.no_grad():
        logits = reference_model(masked_fens, themes, ratings)
        reference_loss = reference_model.elbo_loss(t, logits, fens, masked_fens)

    # compute the integral and get the group structure back
    model_loss = (quadrature_weight * model_loss.reshape(batch_size, group_size, n_quadrature)).sum(dim=2)
    reference_loss = (quadrature_weight * reference_loss.reshape(batch_size, group_size, n_quadrature)).sum(dim=2)
    rewards = rewards.reshape(batch_size, group_size)
    rho = torch.exp(1 / sequence_length * (model_loss - reference_loss))

    # (batch_size,)
    advantages = (rewards - rewards.mean(dim=1, keepdim=True)).to(device)  # Dr GRPO (do not normalize by the standard deviation) https://arxiv.org/pdf/2503.20783
    loss = torch.minimum(rho * advantages, torch.clamp(rho, 1 - eps, 1 + eps) * advantages).mean(dim=1) - beta * kl_estimate(model_loss, reference_loss)  # (batch_size,)
    return -loss  # maximize the loss above

def espo_loss_basic(model, reference_model, fens, themes, ratings, rewards, group_size, eps=0.2, beta=0.1):
    n_samples, sequence_length = fens.shape  # fens.shape == (batch_size * group_size, sequence_length)  batch_size groups of size group_size
    assert n_samples % group_size == 0
    batch_size = n_samples // group_size

    device = ratings.device
    config = model.config

    t = ((torch.rand(1) + torch.arange(n_samples) / n_samples) % 1).to(device)
    alpha_t = config.masking_schedule(t)

    random_mask = torch.rand(fens.size(), device=device) < alpha_t.unsqueeze(1)
    masked_fens = torch.where(random_mask, fens, config.mask_token)

    # use the same mask for both models (antithetic sampling from https://arxiv.org/pdf/2512.03759)
    logits = model(masked_fens, themes, ratings)
    model_loss = model.elbo_loss(t, logits, fens, masked_fens)
    with torch.no_grad():
        logits = reference_model(masked_fens, themes, ratings)
        reference_loss = reference_model.elbo_loss(t, logits, fens, masked_fens)

    # get the group structure back
    model_loss = model_loss.reshape(batch_size, group_size)
    reference_loss = reference_loss.reshape(batch_size, group_size)
    rewards = rewards.reshape(batch_size, group_size)

    advantages = (rewards - rewards.mean(dim=1, keepdim=True)).to(device)
    rho = torch.exp(1 / sequence_length * (model_loss - reference_loss))
    loss = torch.minimum(rho * advantages, torch.clamp(rho, 1 - eps, 1 + eps) * advantages).mean(dim=1) - beta * kl_estimate(model_loss, reference_loss)
    return -loss  # maximize the loss above

def generate_grouped_positions(model, themes, ratings, group_size, steps=256):
    themes = themes.repeat_interleave(group_size, dim=0)
    ratings = ratings.repeat_interleave(group_size, dim=0)

    fens = model.sample(themes, ratings, steps=steps)
    return fens, themes, ratings

state_of_game_tokens = ("opening", "middlegame", "endgame")
endgames = ("pawnEndgame", "bishopEndgame", "knightEndgame", "rookEndgame", "queenEndgame", "queenRookEndgame")

is_mate = "mate"
mate_lengths = ("mateIn1", "mateIn2", "mateIn3", "mateIn4", "mateIn5")
types_of_mate = ("backRankMate", "bodenMate", "smotheredMate", "hookMate", "doubleBishopMate", "arabianMate", "dovetailMate", "anastasiaMate")

lengths = ("oneMove", "short", "long", "veryLong")

winnings = ("crushing", "advantage")
other = ("hangingPiece", "fork", "interference", "kingsideAttack", "zugzwang", "exposedKing", "skewer", "pin", "quietMove", "discoveredAttack", "sacrifice", "deflection", "advancedPawn", "attraction", "promotion", "queensideAttack", "defensiveMove", "attackingF2F7", "clearance", "intermezzo", "equality", "trappedPiece", "xRayAttack", "capturingDefender", "doubleCheck", "enPassant", "castling", "underPromotion")

def generate_random_themes(batch_size):
    themes = []
    for _ in range(batch_size):
        position_themes = [random.choice(lengths)]
        state_of_game = random.choice(state_of_game_tokens)
        position_themes.append(state_of_game)

        if state_of_game == "endgame":
            position_themes.append(random.choice(endgames))
        
        if torch.rand(1) < 0.2:  # about 20% of positions should be mates
            position_themes.append(is_mate)
            position_themes.append(random.choice(mate_lengths))
            position_themes.append(random.choice(types_of_mate))
        else:
            position_themes.append(random.choice(winnings))
            position_themes.append(random.choice(other))

        themes.append(position_themes)
    
    ratings = 3000 * torch.rand((batch_size,)) + 300

    return themes, ratings
