import torch

from typing import Tuple

from .MaskedDiffusion import MaskedDiffusion
from .utils import Position
from .tokenization import theme_preprocessor, scale_ratings, tokens_to_fen, tokens_to_move
from .metrics.model import TagKind


def prepare_input(themes: list[TagKind], rating: float, model: MaskedDiffusion, n_attempts: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepares the input for the MaskedDiffusion model
    """
    device = next(model.parameters()).device
    themes_one_hot = torch.from_numpy(theme_preprocessor.transform([themes])).to(device=device, dtype=torch.float32)
    scaled_ratings = scale_ratings(torch.tensor(rating)).to(device=device, dtype=torch.float32)

    themes_one_hot = themes_one_hot.repeat(n_attempts, 1)
    scaled_ratings = scaled_ratings.repeat(n_attempts)
    return themes_one_hot, scaled_ratings

def generate(themes: list[TagKind], rating: float, model: MaskedDiffusion, n_attempts: int = 256, steps: int = 16) -> list[Position]:
    themes_one_hot, scaled_ratings = prepare_input(themes, rating, model, n_attempts)

    tokens = model.sample(theme_tokens=themes_one_hot, ratings=scaled_ratings, batch_size=n_attempts, steps=steps, generate_move_last=False)
    tokens_cpu = tokens.cpu()
    if model.config.predict_moves:
        fen_tokens = tokens_cpu[:, :model.config.fen_length]
        move_tokens = tokens_cpu[:, model.config.fen_length:]
    else:
        fen_tokens = tokens_cpu
        move_tokens = None
    
    positions = []
    for i in range(n_attempts):
        try:
            fen = tokens_to_fen(fen_tokens[i])
            move = tokens_to_move(move_tokens[i]) if move_tokens is not None else None
            positions.append(Position(fen=fen, move=move, base_rating=rating, base_themes=themes))
        except:
            continue
    
    return positions
