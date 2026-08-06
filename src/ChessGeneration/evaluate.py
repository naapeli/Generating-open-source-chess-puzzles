from chess.engine import SimpleEngine

from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

from .metrics import cook, theme_reward, legal, uniqueness, counter_intuitive, get_unique_puzzle_from_fen
from .utils import Position, Evaluation, EvaluationFlag


_local = threading.local()


class LocalEngine:
    def __init__(self, stockfish_path: Path):
        self.engine = SimpleEngine.popen_uci(stockfish_path)
        self.engine.configure({"Threads": 1})

    def __del__(self):
        try:
            self.engine.quit()
        except Exception:
            pass


def evaluate_position(position: Position, engine: SimpleEngine) -> Evaluation:
    """
    Runs Stockfish to find the puzzle metrics of a single position

    Args:
        position: Position to evaluate
        engine: Stockfish engine

    Returns:
        Evaluation of the position
    """
    is_legal = legal(position.fen)
    unique_solution = False
    counter_intuitive_solution = False
    counter_intuitive_value = 0
    themes_match = False
    mainline = []
    if is_legal:
        engine.configure({"Clear Hash": None})
        counter_intuitive_solution, counter_intuitive_value = counter_intuitive(position.fen, engine, return_value=True)
        unique_solution = uniqueness(position.fen, engine)
        puzzle = get_unique_puzzle_from_fen(position.fen, engine)
        if puzzle:
            actual_themes = cook(puzzle, engine)
            themes_match = theme_reward(position.base_themes, actual_themes)
            mainline = [node.san() for node in puzzle.mainline]

    return Evaluation(
        position=position,
        legal=is_legal,
        unique_solution=unique_solution,
        counter_intuitive_solution=counter_intuitive_solution,
        counter_intuitive_value=counter_intuitive_value,
        themes_match=themes_match,
        mainline=mainline
    )


def evaluate_positions(positions: list[Position], n_jobs: int = 1, stockfish_path: Path = Path("./Stockfish/src/stockfish")) -> list[Evaluation]:
    """
    Evaluates a list of positions in parallel.

    Args:
        positions: List of positions to evaluate.
        n_jobs: Number of threads to use.
        stockfish_path: Path to the Stockfish executable.

    Returns:
        List of Evaluations matching the order of the input positions.
    """
    def _init():
        _local.engine = LocalEngine(stockfish_path)

    def _eval(pos):
        return evaluate_position(pos, _local.engine.engine)

    with ThreadPoolExecutor(max_workers=n_jobs, initializer=_init) as executor:
        return list(executor.map(_eval, positions))

def choose_best_evaluations(evaluations: list[Evaluation], n: int = 3) -> list[Evaluation]:
    """
    Chooses the best n evaluations from a list of evaluations

    Args:
        evaluations: List of evaluations to choose from
        n: Number of evaluations to choose

    Returns:
        List of best n evaluations
    """
    if n > len(evaluations): raise ValueError(f"n must be less than or equal to the number of evaluations")
    sorted_evals = sorted(
        evaluations,
        key=lambda e: (e.legal, e.unique_solution, e.themes_match, e.counter_intuitive_value),
        reverse=True
    )
    return sorted_evals[:n]


def is_evaluation_good_enough(evaluation: Evaluation, flag: EvaluationFlag = EvaluationFlag.UNIQUE_AND_THEMES) -> bool:
    """
    Checks if a position is good enough or if more positions need to be sampled
    """
    if flag == EvaluationFlag.UNIQUE_SOLUTION:
        return evaluation.unique_solution
    elif flag == EvaluationFlag.UNIQUE_AND_THEMES:
        return evaluation.unique_solution and evaluation.themes_match
    elif flag == EvaluationFlag.UNIQUE_AND_THEMES_AND_COUNTER_INTUITIVE:
        return evaluation.unique_solution and evaluation.themes_match and evaluation.counter_intuitive_solution
