from dataclasses import dataclass, field

from .MaskingSchedule import string_to_schedule


@dataclass
class Config:
    schedule: str = "linear"

    # tokenization
    n_fen_tokens: int = 48
    n_move_tokens: int = 4
    n_themes: int = 66
    rating_dim: int = 1
    fen_length: int = 76
    move_length: int = 5
    mask_token: int = field(init=False)
    predict_moves: bool = True
    use_context: bool = True
    n_tokens: int = field(init=False)

    def __post_init__(self):
        self.n_tokens = self.n_fen_tokens + (self.n_move_tokens if self.predict_moves else 0)
        self.mask_token = self.n_tokens
        self.masking_schedule = string_to_schedule(self.schedule)
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__post_init__()

    # model architecture
    n_heads: int = 8
    n_layers: int = 16
    embed_dim: int = 1024

    # optimizer and training
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 1024
    n_steps: int = 100_000
    validation_interval: int = 1000
    train_logging_interval: int = 100
    save_interval: int = 10_000
    n_validation_generations: int = 1
