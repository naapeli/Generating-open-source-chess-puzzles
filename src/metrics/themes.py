from typing import Optional, List
import os
from time import perf_counter
from copy import deepcopy

import chess
from chess import Color, Move
from chess.engine import Mate, Limit, SimpleEngine, Cp
import chess.pgn
from chess.pgn import ChildNode

from metrics.model import Puzzle, NextMovePair
from metrics.util import win_chances, count_mates, get_next_move_pair
from metrics.cook import cook


mate_soon = Mate(15)
# pair_limit = Limit(depth=50, time=30, nodes=25_000_000)
# mate_defense_limit = Limit(depth=15, time=10, nodes=8_000_000)

pair_limit = Limit(depth=15, time=10, nodes=8_000_000)
mate_defense_limit = Limit(depth=8, time=5, nodes=4_000_000)

TAU_UNI = 0.5
TAU_CNT = 0.1

def legal(fen):
    try:
        board = chess.Board(fen)
        return board.is_valid()
    except ValueError:
        return False

def get_unique_puzzle_from_fen(fen, engine: SimpleEngine):
    board = chess.Board(fen)
    if board.is_game_over(): return None  # NOTE: just check that the model has not generated a position that is checkmate already
    if board.legal_moves.count() == 1: return None  # NOTE: had a problem in this position without this: 8/8/p7/P7/1P6/6pk/6p1/7K w - - 0 52
    game = chess.pgn.Game.from_board(board)
    info = engine.analyse(board, limit=pair_limit)
    score = info["score"].pov(board.turn)

    if score > mate_soon:
        mate_solution = cook_mate(game, board.turn, engine)
        if mate_solution is None:
            return None
        return Puzzle(game, score)
    else:
        solution = cook_advantage(deepcopy(game), board.turn, engine)
        if not solution:
            return None
        
        # make sure the solution is odd, the last move is not the only one available and the puzzle is not a one mover
        while len(solution) % 2 == 0 or not solution[-1].second:
            solution = solution[:-1]
        if not solution or len(solution) == 1:
            return None
        
        # update the main line
        node = game
        for pair in solution:
            move = pair.best.move
            node = node.add_main_variation(move)

        return Puzzle(game, score)

def counter_intuitive(fen, engine: SimpleEngine):
    board = chess.Board(fen)
    if board.is_game_over(): return False  # NOTE: just check that the model has not generated a position that is checkmate already
    history = []        
    with engine.analysis(board, pair_limit) as analysis:
        for info in analysis:
            if "pv" in info and "depth" in info:
                move_depth = info["depth"]
                best_move = info["pv"][0]
                history.append((move_depth, best_move))

    critical_point = pair_limit.depth
    
    final_best_move = history[-1][1]
    for move_depth, move in history:
        if move == final_best_move:
            critical_point = move_depth
            break

    v_critical_point = critical_point - 1

    v_capture_material = 0
    if board.is_capture(final_best_move):
        if board.is_en_passant(final_best_move):
            captured_value = 1
        else:
            captured_piece = board.piece_at(final_best_move.to_square)
            values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
            captured_value = values.get(captured_piece.piece_type, 0)
        
        v_capture_material = -captured_value / 9

    score = (0.8 * v_critical_point) + (0.1 * v_capture_material)
    return score > TAU_CNT

def is_valid_attack(pair: NextMovePair, engine: SimpleEngine) -> bool:
    return (
        pair.second is None or
        is_valid_mate_in_one(pair, engine) or
        win_chances(pair.best.score) - win_chances(pair.second.score) > + TAU_UNI
    )

def is_valid_mate_in_one(pair: NextMovePair, engine: SimpleEngine) -> bool:
        if pair.best.score != Mate(1):
            return False
        non_mate_win_threshold = 0.6
        if not pair.second or win_chances(pair.second.score) <= non_mate_win_threshold:
            return True
        if pair.second.score == Mate(1):
            # if there's more than one mate in one, gotta look if the best non-mating move is bad enough
            mates = count_mates(pair.node.board())
            info = engine.analyse(pair.node.board(), multipv=mates + 1, limit=pair_limit)
            scores =  [pv["score"].pov(pair.winner) for pv in info]
            # the first non-matein1 move is the last element
            if scores[-1] < Mate(1) and win_chances(scores[-1]) > non_mate_win_threshold:
                    return False
            return True
        return False

def get_next_pair(node: ChildNode, winner: Color, engine: SimpleEngine) -> Optional[NextMovePair]:
        pair = get_next_move_pair(engine, node, winner, pair_limit)
        if node.board().turn == winner and not is_valid_attack(pair, engine):
            return None
        return pair

def get_next_move(node: ChildNode, engine: SimpleEngine, limit: Limit) -> Optional[Move]:
        result = engine.play(node.board(), limit = limit)
        return result.move if result else None

def cook_mate(node: ChildNode, winner: Color, engine: SimpleEngine) -> Optional[List[Move]]:
    board = node.board()
    if board.is_game_over():
        return []

    if board.turn == winner:
        pair = get_next_pair(node, winner, engine)
        if not pair:
            return None
        if pair.best.score < mate_soon:
            return None
        move = pair.best.move
    else:
        next = get_next_move(node, engine, mate_defense_limit)
        if not next:
            return None
        move = next

    follow_up = cook_mate(node.add_main_variation(move), winner, engine)
    if follow_up is None:
        return None

    return [move] + follow_up


def cook_advantage(node: ChildNode, winner: Color, engine: SimpleEngine) -> Optional[List[NextMovePair]]:
    board = node.board()
    if board.is_game_over():  # if we accidentally find checkmate, make sure we do not get an error
        return []
    if board.is_repetition(2):
        return None

    pair = get_next_pair(node, winner, engine)
    if not pair:
        return []
    if pair.best.score < Cp(200):  # should maybe remove if we follow the paper exactly (I think paper does not do this, but Lichess Puzzler does)
        return None

    follow_up = cook_advantage(node.add_main_variation(pair.best.move), winner, engine)
    if follow_up is None:
        return None

    return [pair] + follow_up

if __name__ == "__main__":
    stockfish_path = "./Stockfish/src/stockfish"
    engine = SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Threads": os.cpu_count()})
    engine.configure({"Hash": 4096})

    start = perf_counter()
    puzzle = get_unique_puzzle_from_fen("1r2k1r1/pbppnp1p/1b3P2/8/Q7/B1PB1q2/P4PPP/3R2K1 w - - 1 0", engine)
    print(puzzle.game)
    print(cook(puzzle, engine))  # first mate-in-4 in https://chess.stackexchange.com/questions/19633/chess-problem-database-with-pgn-or-fen

    engine.configure({"Clear Hash": None})
    puzzle = get_unique_puzzle_from_fen("q5nr/1ppknQpp/3p4/1P2p3/4P3/B1PP1b2/B5PP/5K2 w - - 1 18", engine)
    print(puzzle.game)
    print(cook(puzzle, engine))  # [mate mateIn2 middlegame short]

    engine.configure({"Clear Hash": None})
    puzzle = get_unique_puzzle_from_fen("r3r1k1/p4ppp/2p2n2/1p6/3P1qb1/2NQ2R1/PPB2PP1/R1B3K1 b - - 6 18", engine)
    print(puzzle.game)
    print(cook(puzzle, engine))  # [advantage attraction fork middlegame sacrifice veryLong]

    engine.configure({"Clear Hash": None})
    puzzle = get_unique_puzzle_from_fen("8/8/1pp1k1p1/4Pp1p/2PK1P1P/pP6/P7/8 w - - 2 36", engine)
    print(puzzle.game)
    print(cook(puzzle, engine))  # [crushing endgame long pawnEndgame quietMove zugzwang]

    print("Time taken:", perf_counter() - start)

    engine.quit()
