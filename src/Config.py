from dataclasses import dataclass

from MaskingSchedule.MaskingSchedule import string_to_schedule
from tokenization.tokenization import FENTokens


@dataclass
class Config:
    schedule: str = "cosine"
    masking_schedule = string_to_schedule(schedule)

    # tokenization
    n_fen_tokens: int = 48
    n_move_tokens: int = 4
    n_themes: int = 66
    rating_dim: int = 1
    fen_length: int = 76
    move_length: int = 5
    mask_token: FENTokens = FENTokens.mask
    predict_moves: bool = True
    use_context: bool = True
    n_tokens = n_fen_tokens + n_move_tokens if predict_moves else n_fen_tokens

    # model architecture
    n_heads: int = 8
    n_layers: int = 16
    embed_dim: int = 1024

    # optimizer and training
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 1024
    n_steps: int = 100_000
    validation_interval: int = 1000
    train_logging_interval: int = 100
    save_interval: int = 10_000
    n_validation_generations: int = 1
