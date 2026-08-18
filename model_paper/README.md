---
license: mit
tags:
- diffusers
- chess
- custom-pipeline
pipeline_tag: text-generation
---

# Chess Puzzle Generator

A masked diffusion model for generating chess puzzles conditioned on themes, ratings, moves and partial boards. This model is not guaranteed to generate a puzzle and the generations should be filtered afterwards.

## Models

We provide two models. The main model, which is an updated version of the one presented in our paper, is trained mainly for generating as many positions with a unique solution that match the themes the user asked for. In contrast, the model in the paper was mainly trained to maximize counter-intuitivity and uniqueness instead of thematic accuracy. The model from our paper can be used with the revision="paper" parameter.

## Pipeline Documentation

### `ChessPuzzlePipeline.__call__`

```python
pipeline(
    themes: str | list[str | PuzzleTheme] | PuzzleTheme = None,
    rating: float = 1500.0,
    partial_board: str = None,
    best_move: str = None,
    batch_size: int = 1,
    steps: int = 256,
    temperature: float = 1.0,
    schedule: str | Schedule = Schedule.linear,
    generate_move_last: bool = True,
) -> list[Position]
```

### Parameters

- **`themes`** (`str | list[str | PuzzleTheme] | PuzzleTheme`, optional, default: `None`):  
  The thematic tags to condition the puzzle generation on. Supports:
  - Space-separated string: `"mateIn2 middlegame"`
  - List of strings: `["mateIn2", "middlegame"]`
  - List of `Theme` enum members: `[pipeline.Theme.mateIn2, pipeline.Theme.middlegame]`
  - Single `Theme` enum member: `pipeline.Theme.mateIn1`

- **`rating`** (`float`, optional, default: `1500.0`):  
  Target puzzle difficulty rating. Scaled based on Lichess puzzle ratings (range: 399 to 3395).

- **`partial_board`** (`str`, optional, default: `None`):  
  A partial FEN string to condition on, where unknown squares/fields are represented with `?`.  
  *Example:* `"?????rk?/?????ppp/????????/????????/????????/???B????/????????/???????? w ??-- - ? ?"`

- **`best_move`** (`str`, optional, default: `None`):  
  A UCI-format move string to force as the solution (e.g. `"d3h7"`, `"e7e8q"`, `"e2??"`). Can also contain `?` for unknown characters.

- **`batch_size`** (`int`, optional, default: `1`):  
  Number of puzzle positions to generate in parallel.

- **`steps`** (`int`, optional, default: `256`):  
  Number of discrete diffusion unmasking steps. Higher steps generally yield higher quality and more valid positions, but lower values work as well. Tested values between 16 and 256.

- **`temperature`** (`float`, optional, default: `1.0`):  
  Sampling temperature applied to the unmasking logits. Lower values make sampling more greedy/deterministic.

- **`schedule`** (`str | Schedule`, optional, default: `Schedule.linear`):  
  Noise schedule used for unmasking tokens. Can be a string or `pipeline.Schedule` enum:
  - `pipeline.Schedule.linear` (`"linear"`)
  - `pipeline.Schedule.cosine` (`"cosine"`)
  - `pipeline.Schedule.geometric` (`"geometric"`)
  - `pipeline.Schedule.polynomial` (`"polynomial"`)

- **`generate_move_last`** (`bool`, optional, default: `True`):  
  When `True`, the model first generates the full 64-square board position across `steps`, and then unmasks the 5 solution move tokens in a subsequent phase.

### Return Value

Returns a `list[Position]` of length `batch_size`, where each `Position` is a dataclass:
```python
@dataclass
class Position:
    fen: str
    move: str | None
```

You can access the generated FEN and move directly as attributes:
```python
position = results[0]
print(position.fen)   # "2nrb3/n2k2qp/1ppp4/4p3/5P2/R7/1PPBP1PP/R4NK1 w - - 0 22"
print(position.move)  # "a3a7"
```

---

### Available Themes

All 66 supported themes can be accessed via `pipeline.Theme.<name>` or passed as strings:

| Category | Available Themes |
| :--- | :--- |
| **State-of-game** | `opening`, `middlegame`, `endgame` |
| **Type-of-endgame** | `pawnEndgame`, `bishopEndgame`, `knightEndgame`, `rookEndgame`, `queenEndgame`, `queenRookEndgame` |
| **Type-of-checkmate** | `mate`, `backRankMate`, `bodenMate`, `smotheredMate`, `hookMate`, `doubleBishopMate`, `arabianMate`, `dovetailMate`, `anastasiaMate`, `triangleMate`, `balestraMate`, `killBoxMate`, `blindSwineMate`, `cornerMate`, `vukovicMate` |
| **Length-of-checkmate** | `mateIn1`, `mateIn2`, `mateIn3`, `mateIn4`, `mateIn5` |
| **Length-of-puzzle** | `oneMove`, `short`, `long`, `veryLong` |
| **Winning** | `crushing`, `advantage` |
| **Other** | `hangingPiece`, `fork`, `interference`, `kingsideAttack`, `zugzwang`, `exposedKing`, `skewer`, `pin`, `quietMove`, `discoveredAttack`, `sacrifice`, `deflection`, `advancedPawn`, `attraction`, `promotion`, `queensideAttack`, `defensiveMove`, `attackingF2F7`, `clearance`, `intermezzo`, `equality`, `trappedPiece`, `xRayAttack`, `capturingDefender`, `doubleCheck`, `enPassant`, `castling`, `underPromotion`, `master`, `masterVsMaster`, `superGM` |

---

## Usage

```python
import torch
from diffusers import DiffusionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

pipeline = DiffusionPipeline.from_pretrained(
    "naapeli/chess-puzzle-generator",
    trust_remote_code=True,
)
pipeline.to(device)

# For exactly the same model as in the paper, use revision="paper":
# pipeline = DiffusionPipeline.from_pretrained(
#     "naapeli/chess-puzzle-generator",
#     revision="paper",
#     trust_remote_code=True,
# )
# pipeline.to(device)

themes = pipeline.Theme
schedules = pipeline.Schedule

# 1. Unconditional generation conditioned on themes and rating
results = pipeline(
    themes=[themes.mateIn2, themes.middlegame],
    rating=1800,
    batch_size=1,
    steps=64,
    schedule=schedules.linear,
)
print(results[0].fen, results[0].move)

# 2. Condition on a partial board and a best move
partial_fen = "?????rk?/?????ppp/????????/????????/????????/???B????/????????/???????? w ??-- - ? ?"
best_move = "d3h7"
results = pipeline(
    themes=themes.mate,
    rating=1600,
    partial_board=partial_fen,
    best_move=best_move,
    batch_size=1,
    steps=256,
    schedule=schedules.cosine,
)
print(results[0].fen, results[0].move)
```
