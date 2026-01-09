from dataclasses import dataclass

from MaskingSchedule.MaskingSchedule import MaskingSchedule, CosineSchedule
from tokenization.tokenization import FENTokens


@dataclass
class Config:
    masking_schedule: MaskingSchedule = CosineSchedule()

    # tokenization
    n_fen_tokens: int = 48  # 48 real tokens and one mask token
    n_themes: int = 66
    rating_dim: int = 1
    fen_length: int = 76
    mask_token: FENTokens = FENTokens.mask

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
    n_validation_generations: int = 10
