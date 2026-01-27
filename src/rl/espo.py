import torch
import numpy as np
import pandas as pd

import random


n_quadrature = 2
t_points, quadrature_weights = np.polynomial.legendre.leggauss(n_quadrature)  # estimate the integral with a gaussian quadrature (https://arxiv.org/pdf/2510.08554)
t_points, quadrature_weights = torch.from_numpy(t_points), torch.from_numpy(quadrature_weights)
t_points, quadrature_weights = (1 - 0) / 2 * t_points + (1 + 0) / 2, (1 - 0) / 2 * quadrature_weights  # https://en.wikipedia.org/wiki/Gaussian_quadrature#Change_of_interval


def kl_estimate(model_elbo, reference_elbo):  # k2 from http://joschu.net/blog/kl-approx.html and https://arxiv.org/pdf/2512.03759 (biased, but low variance estimator)
    return 0.5 * torch.mean((model_elbo - reference_elbo) ** 2)

def espo_loss(model, reference_model, fens, themes, ratings, rewards, group_size, eps=0.2, beta=1):
    n_samples, sequence_length = fens.shape
    assert n_samples % group_size == 0
    n_groups = n_samples // group_size

    device = ratings.device
    config = model.config

    t = t_points[None, :].expand(n_samples, n_quadrature)
    weights = quadrature_weights[None, :].expand(2 * n_samples, n_quadrature)

    alpha_t = config.masking_schedule(t).to(device)
    random_mask = torch.rand(fens.size(), device=device) < alpha_t

    # use the same mask for both models (antithetic sampling from https://arxiv.org/pdf/2512.03759)
    antithetic_sampling_t = torch.concat([t, 1 - t], dim=0)
    masked_fens = torch.concat([torch.where(random_mask, fens, config.mask_token), torch.where(~random_mask, fens, config.mask_token)], dim=0)
    target_fens = torch.concat([fens, fens], dim=0)
    antithetic_sampling_themes = torch.concat([themes, themes], dim=0)
    antithetic_sampling_ratings = torch.concat([ratings, ratings], dim=0)

    # use coupled sampling, i.e. use the same masks for both models (https://arxiv.org/pdf/2512.03759)
    model_loss = model.variance_reduced_elbo(masked_fens, target_fens, antithetic_sampling_themes, antithetic_sampling_ratings, antithetic_sampling_t, quadrature_weights=weights)  # (2 * group_size * n_groups,)
    with torch.no_grad():
        reference_loss = reference_model.variance_reduced_elbo(masked_fens, target_fens, antithetic_sampling_themes, antithetic_sampling_ratings, antithetic_sampling_t, quadrature_weights=weights)  # (2 * group_size * n_groups,)

    # get the group structure back
    model_loss = model_loss.reshape(2 * n_groups, group_size)
    reference_loss = reference_loss.reshape(2 * n_groups, group_size)
    rewards = rewards.reshape(2 * n_groups, group_size)

    advantages = rewards - rewards.mean(dim=1, keepdim=True)  # Dr GRPO (do not normalize by the standard deviation) https://arxiv.org/pdf/2503.20783
    rho = torch.exp(1 / sequence_length * (model_loss - reference_loss))
    loss = torch.minimum(rho * advantages, torch.clamp(rho, 1 - eps, 1 + eps) * advantages).mean(dim=1) - beta * kl_estimate(model_loss, reference_loss)  # (2 * n_groups,)
    return loss

def generate_grouped_positions(model, batch_size, group_size, theme_preprocessor, scale_ratings, steps=256):
    themes, ratings = generate_random_themes(theme_preprocessor, scale_ratings, batch_size)
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
other = ("hangingPiece", "fork", "interference", "kingsideAttack", "zugzwang", "exposedKing", "skewer", "pin", "quietMove", "discoveredAttack", "sacrifice", "deflection", "advancedPawn", "attraction", "promotion", "superGM", "queensideAttack", "defensiveMove", "attackingF2F7", "clearance", "intermezzo", "equality", "trappedPiece", "xRayAttack", "capturingDefender", "doubleCheck", "enPassant", "castling", "underPromotion")

def generate_random_themes(theme_preprocessor, scale_ratings, batch_size):
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
    
    themes = theme_preprocessor.transform(themes)
    themes = torch.from_numpy(themes)
    ratings = scale_ratings(3000 * torch.rand((batch_size,)) + 300)

    return themes, ratings
