from chess.engine import SimpleEngine

from .metrics import cook, theme_reward, legal, uniqueness, counter_intuitive, get_unique_puzzle_from_fen
from .utils import Position, Evaluation, EvaluationFlag


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
    if is_legal:
        engine.configure({"Clear Hash": None})
        counter_intuitive_solution, counter_intuitive_value = counter_intuitive(position.fen, engine, return_value=True)
        unique_solution = uniqueness(position.fen, engine)
        puzzle = get_unique_puzzle_from_fen(position.fen, engine)
        if puzzle:
            actual_themes = cook(puzzle, engine)
            themes_match = theme_reward(position.base_themes, actual_themes)

    return Evaluation(
        position=position,
        legal=is_legal,
        unique_solution=unique_solution,
        counter_intuitive_solution=counter_intuitive_solution,
        counter_intuitive_value=counter_intuitive_value,
        themes_match=themes_match
    )

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


def evaluation_good_enough(evaluation: Evaluation, flag: EvaluationFlag = EvaluationFlag.UNIQUE_AND_THEMES_AND_COUNTER_INTUITIVE) -> bool:
    """
    Checks if a position is good enough or if more positions need to be sampled
    """
    if flag == EvaluationFlag.UNIQUE_SOLUTION:
        return evaluation.unique_solution
    elif flag == EvaluationFlag.UNIQUE_AND_THEMES:
        return evaluation.unique_solution and evaluation.themes_match
    elif flag == EvaluationFlag.UNIQUE_AND_THEMES_AND_COUNTER_INTUITIVE:
        return evaluation.unique_solution and evaluation.themes_match and evaluation.counter_intuitive_solution
