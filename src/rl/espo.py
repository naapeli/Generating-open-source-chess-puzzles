import torch
import numpy as np

import random


n_quadrature = 3
t_points, quadrature_weights = np.polynomial.legendre.leggauss(n_quadrature)  # estimate the integral with a gaussian quadrature (https://arxiv.org/pdf/2510.08554)
t_points, quadrature_weights = torch.from_numpy(t_points), torch.from_numpy(quadrature_weights)
t_points, quadrature_weights = (1 - 0) / 2 * t_points + (1 + 0) / 2, (1 - 0) / 2 * quadrature_weights  # https://en.wikipedia.org/wiki/Gaussian_quadrature#Change_of_interval


def kl_estimate(model_elbo, reference_elbo):
    # k2 from http://joschu.net/blog/kl-approx.html and https://arxiv.org/pdf/2512.03759 (biased, but low variance estimator)
    return 0.5 * torch.mean((model_elbo - reference_elbo) ** 2, dim=1)

def compute_elbo(model, fens, themes, ratings, mask=None, return_mask=False):
    device = ratings.device
    n_samples = len(ratings)

    config = model.module.config if hasattr(model, "module") else model.config
    module = model.module if hasattr(model, "module") else model

    t = t_points.repeat(n_samples).to(device)
    alpha_t = config.masking_schedule(t)
    quadrature_weight = quadrature_weights.unsqueeze(0).to(device)

    fens = fens.repeat_interleave(n_quadrature, dim=0)
    themes = themes.repeat_interleave(n_quadrature, dim=0)
    ratings = ratings.repeat_interleave(n_quadrature, dim=0)

    random_mask = torch.rand(fens.size(), device=device) < alpha_t.unsqueeze(1) if mask is None else mask
    masked_fens = torch.where(random_mask, fens, config.mask_token)

    logits = model(masked_fens, themes, ratings)
    elbo = module.elbo_loss(t, logits, fens, masked_fens)
    elbo = (quadrature_weight * elbo.reshape(n_samples, n_quadrature)).sum(dim=1)
    elbo = -elbo  # model.elbo_loss returns an upper bound of the negative log likelihood, which we minimized during supervised training
    assert (elbo <= 0).all(), f"elbo should be a lower bound of a probability, {elbo}"
    if return_mask:
        return elbo, random_mask
    return elbo

def compute_elbo_basic(model, fens, themes, ratings, mask=None, return_mask=False):
    device = ratings.device
    n_samples = len(ratings)

    config = model.module.config if hasattr(model, "module") else model.config
    module = model.module if hasattr(model, "module") else model

    t = ((torch.rand(1) + torch.arange(n_samples) / n_samples) % 1).to(device)
    alpha_t = config.masking_schedule(t)

    random_mask = torch.rand(fens.size(), device=device) < alpha_t.unsqueeze(1) if mask is None else mask
    masked_fens = torch.where(random_mask, fens, model.config.mask_token)

    logits = model(masked_fens, themes, ratings)
    elbo = module.elbo_loss(t, logits, fens, masked_fens)
    elbo = -elbo  # model.elbo_loss returns an upper bound of the negative log likelihood, which we minimized during supervised training
    assert (elbo <= 0).all(), f"elbo should be a lower bound of a probability, {elbo}"
    if return_mask:
        return elbo, random_mask
    return elbo

def espo_loss(model, reference_elbos, old_elbos, fens, themes, ratings, rewards, group_size, mask=None, eps=0.2, beta=0.1):
    n_samples, sequence_length = fens.shape  # fens.shape == (batch_size * group_size, sequence_length)  batch_size groups of size group_size
    assert n_samples % group_size == 0
    batch_size = n_samples // group_size

    device = ratings.device
    elbo = compute_elbo(model, fens, themes, ratings, mask=mask, return_mask=False)
    # elbo = compute_elbo_basic(model, fens, themes, ratings, mask=mask, return_mask=False)

    rewards = rewards.reshape(batch_size, group_size)
    elbo = elbo.reshape(batch_size, group_size)
    reference_elbos = reference_elbos.reshape(batch_size, group_size)
    old_elbos = old_elbos.reshape(batch_size, group_size)
    # this assumes that elbo and old_elbos are negative (lower bounds of the log probability)
    rho = torch.exp((elbo - old_elbos) / sequence_length)  # importance sampling (generate from old model, update current model)

    # (batch_size,)
    advantages = (rewards - rewards.mean(dim=1, keepdim=True)).to(device)  # Dr GRPO (do not normalize by the standard deviation) https://arxiv.org/pdf/2503.20783
    loss = torch.minimum(rho * advantages, torch.clamp(rho, 1 - eps, 1 + eps) * advantages).mean(dim=1)
    kl = kl_estimate(elbo, reference_elbos)
    loss = loss - beta * kl
    return -loss, kl  # maximize the loss above

def generate_grouped_positions(model, themes, ratings, group_size, steps=256):
    themes = themes.repeat_interleave(group_size, dim=0)
    ratings = ratings.repeat_interleave(group_size, dim=0)

    module = model.module if hasattr(model, "module") else model

    fens = module.sample(themes, ratings, steps=steps)
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
        
        if torch.rand(1) < 0.1:  # about 20% of positions should be mates
            position_themes.append(is_mate)
            position_themes.append(random.choice(mate_lengths))
            position_themes.append(random.choice(types_of_mate))
        else:
            position_themes.append(random.choice(winnings))
            position_themes.append(random.choice(other))

        themes.append(position_themes)
    
    ratings = 3000 * torch.rand((batch_size,)) + 300

    return themes, ratings

def theme_reward(base_themes, puzzle_themes):
    if base_themes[1] not in puzzle_themes:  # the state of the game must match
        return False
    if base_themes[1] == "endgame" and base_themes[2] not in puzzle_themes:
        return False
    
    # checkmate
    is_mate = "mate" in base_themes
    if is_mate and "mate" not in puzzle_themes:
        return False
    if is_mate and base_themes[base_themes.index("mate") + 1] not in puzzle_themes:  # they must be the same type of checkmate
        return False
    
    # just a winning position
    if not is_mate and base_themes[-1] not in puzzle_themes:  # the other component in a not mating position must match
        return False

    return True
