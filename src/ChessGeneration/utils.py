from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


@dataclass(frozen=True)
class Position:
    fen: str
    move: Optional[str]
    base_rating: float
    base_themes: list[str]

@dataclass(frozen=True)
class Evaluation:
    position: Position
    legal: bool
    unique_solution: bool
    counter_intuitive_solution: bool
    counter_intuitive_value: float
    themes_match: bool
    actual_themes: list[str] = field(default_factory=list)
    mainline: list[str] = field(default_factory=list)

class EvaluationFlag(Enum):
    UNIQUE_SOLUTION = auto()
    UNIQUE_AND_THEMES = auto()
    UNIQUE_AND_THEMES_AND_COUNTER_INTUITIVE = auto()
