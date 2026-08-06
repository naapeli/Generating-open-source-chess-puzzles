import os
import pytest
from chess.engine import SimpleEngine

from ChessGeneration import Position, Evaluation, EvaluationFlag, evaluate_position, evaluate_positions, choose_best_evaluations, is_evaluation_good_enough


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
    assert is_evaluation_good_enough(e_all, EvaluationFlag.UNIQUE_SOLUTION) is True
    assert is_evaluation_good_enough(e_no_unique, EvaluationFlag.UNIQUE_SOLUTION) is False

    # UNIQUE_AND_THEMES flag
    assert is_evaluation_good_enough(e_all, EvaluationFlag.UNIQUE_AND_THEMES) is True
    assert is_evaluation_good_enough(e_no_themes, EvaluationFlag.UNIQUE_AND_THEMES) is False

    # UNIQUE_AND_THEMES_AND_COUNTER_INTUITIVE flag
    assert is_evaluation_good_enough(e_all, EvaluationFlag.UNIQUE_AND_THEMES_AND_COUNTER_INTUITIVE) is True
    assert is_evaluation_good_enough(e_no_counter, EvaluationFlag.UNIQUE_AND_THEMES_AND_COUNTER_INTUITIVE) is False


def test_evaluate_position():
    stockfish_path = "./Stockfish/src/stockfish"
    assert os.path.exists(stockfish_path), f"Stockfish binary not found at {stockfish_path}"
    
    engine = SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Threads": 1})
    
    try:
        pos = Position(
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            move="e2e4",
            base_rating=1500.0,
            base_themes=["opening"]
        )
        
        evaluation = evaluate_position(pos, engine)
        
        assert evaluation.position == pos
        assert evaluation.legal is True
        assert isinstance(evaluation.unique_solution, bool)
        assert isinstance(evaluation.counter_intuitive_solution, bool)
        assert isinstance(evaluation.counter_intuitive_value, float)
        assert isinstance(evaluation.themes_match, bool)
        assert isinstance(evaluation.mainline, list)
    finally:
        engine.quit()


def test_evaluate_positions():
    stockfish_path = "./Stockfish/src/stockfish"
    assert os.path.exists(stockfish_path), f"Stockfish binary not found at {stockfish_path}"

    # pos1 = Position(
    #     fen="3r2k1/1p4p1/p1p1P1r1/2bqn3/3PnN2/3B1pPp/PP5P/RQ1R2K1 w - - 4 29",
    #     move="-",
    #     base_rating=-1,
    #     base_themes=["long", "middlegame"]
    # )
    pos1 = Position(
        fen="3r3r/pR1nkp2/4p1p1/P1P5/8/2n5/5PPQ/5RK1 b - - 0 30",
        move="-",
        base_rating=-1,
        base_themes=["long", "middlegame", "mate", "attraction", "anastasiaMate"]
    )
    pos2 = Position(
        fen="2r1n1k1/p4ppp/1p2p3/6q1/8/3QP2P/4BPPK/R3N3 b - - 2 39",
        move="-",
        base_rating=-1,
        base_themes=["middlegame", "advantage", "short", "fork"]
    )


    # Run with 2 threads
    evaluations = evaluate_positions([pos1, pos2], n_jobs=2, stockfish_path=stockfish_path)

    assert len(evaluations) == 2
    assert evaluations[0].position == pos1
    assert evaluations[0].legal is True
    assert evaluations[0].unique_solution is True
    assert evaluations[0].themes_match is True
    assert evaluations[0].counter_intuitive_solution is True
    assert isinstance(evaluations[0].mainline, list)
    assert len(evaluations[0].mainline) > 0
    assert evaluations[0].mainline[0] == "Ne2+"

    assert evaluations[1].position == pos2
    assert evaluations[1].legal is True
    assert evaluations[1].unique_solution is True
    assert evaluations[1].themes_match is True
    assert evaluations[1].counter_intuitive_solution is False
    assert isinstance(evaluations[1].mainline, list)
    assert len(evaluations[1].mainline) > 0
    assert evaluations[1].mainline[0] == "Qe5+"
