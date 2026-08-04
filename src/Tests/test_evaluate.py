import os
import pytest
from chess.engine import SimpleEngine

from ChessGeneration.utils import Position, Evaluation, EvaluationFlag
from ChessGeneration.evaluate import evaluate_position, choose_best_evaluations, evaluation_good_enough


def test_choose_best_evaluations():
    # Helper to construct dummy Position
    pos = Position(fen="8/8/8/8/8/8/8/8 w - - 0 1", move="e2e4", base_rating=1500, base_themes=["fork"])
    
    # Priority: e.legal, e.unique_solution, e.themes_match, e.counter_intuitive_value
    e1 = Evaluation(position=pos, legal=True, unique_solution=True, counter_intuitive_solution=True, counter_intuitive_value=0.8, themes_match=True)
    e2 = Evaluation(position=pos, legal=True, unique_solution=True, counter_intuitive_solution=True, counter_intuitive_value=0.5, themes_match=True)
    e3 = Evaluation(position=pos, legal=True, unique_solution=True, counter_intuitive_solution=False, counter_intuitive_value=0.2, themes_match=False)
    e4 = Evaluation(position=pos, legal=True, unique_solution=False, counter_intuitive_solution=False, counter_intuitive_value=0.9, themes_match=True)
    e5 = Evaluation(position=pos, legal=False, unique_solution=True, counter_intuitive_solution=True, counter_intuitive_value=0.9, themes_match=True)

    # Let's shuffle them and verify sorting
    evals = [e5, e4, e3, e2, e1]
    best_3 = choose_best_evaluations(evals, n=3)
    
    # Expected order:
    # 1. e1 (legal=True, unique=True, themes=True, value=0.8)
    # 2. e2 (legal=True, unique=True, themes=True, value=0.5)
    # 3. e3 (legal=True, unique=True, themes=False, value=0.2)
    # e4 is 4th because unique=False
    # e5 is 5th because legal=False
    assert best_3[0] == e1
    assert best_3[1] == e2
    assert best_3[2] == e3

    # Error case
    with pytest.raises(ValueError):
        choose_best_evaluations(evals, n=10)


def test_evaluation_good_enough():
    pos = Position(fen="8/8/8/8/8/8/8/8 w - - 0 1", move="e2e4", base_rating=1500, base_themes=["fork"])
    
    e_all = Evaluation(position=pos, legal=True, unique_solution=True, counter_intuitive_solution=True, counter_intuitive_value=0.8, themes_match=True)
    e_no_counter = Evaluation(position=pos, legal=True, unique_solution=True, counter_intuitive_solution=False, counter_intuitive_value=0.0, themes_match=True)
    e_no_themes = Evaluation(position=pos, legal=True, unique_solution=True, counter_intuitive_solution=True, counter_intuitive_value=0.8, themes_match=False)
    e_no_unique = Evaluation(position=pos, legal=True, unique_solution=False, counter_intuitive_solution=True, counter_intuitive_value=0.8, themes_match=True)

    # UNIQUE_SOLUTION flag
    assert evaluation_good_enough(e_all, EvaluationFlag.UNIQUE_SOLUTION) is True
    assert evaluation_good_enough(e_no_unique, EvaluationFlag.UNIQUE_SOLUTION) is False

    # UNIQUE_AND_THEMES flag
    assert evaluation_good_enough(e_all, EvaluationFlag.UNIQUE_AND_THEMES) is True
    assert evaluation_good_enough(e_no_themes, EvaluationFlag.UNIQUE_AND_THEMES) is False

    # UNIQUE_AND_THEMES_AND_COUNTER_INTUITIVE flag
    assert evaluation_good_enough(e_all, EvaluationFlag.UNIQUE_AND_THEMES_AND_COUNTER_INTUITIVE) is True
    assert evaluation_good_enough(e_no_counter, EvaluationFlag.UNIQUE_AND_THEMES_AND_COUNTER_INTUITIVE) is False


def test_evaluate_position():
    # Spawns real Stockfish engine
    stockfish_path = "./Stockfish/src/stockfish"
    assert os.path.exists(stockfish_path), f"Stockfish binary not found at {stockfish_path}"
    
    engine = SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Threads": 1}) # keep it lightweight for tests
    
    try:
        # A known legal position (starting board)
        pos = Position(
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            move="e2e4",
            base_rating=1500.0,
            base_themes=["opening"]
        )
        
        evaluation = evaluate_position(pos, engine)
        
        # Verify correctness of output types and fields
        assert evaluation.position == pos
        assert evaluation.legal is True
        assert isinstance(evaluation.unique_solution, bool)
        assert isinstance(evaluation.counter_intuitive_solution, bool)
        assert isinstance(evaluation.counter_intuitive_value, float)
        assert isinstance(evaluation.themes_match, bool)
    finally:
        engine.quit()
