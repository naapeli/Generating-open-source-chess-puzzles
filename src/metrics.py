import chess
import chess.engine

import math


def calculate_winning_chances(cp):
    # from https://lichess.org/page/accuracy and https://arxiv.org/pdf/2402.04494 and other papers
    MULTIPLIER = 0.00368208
    return 2 / (1 + math.exp(-MULTIPLIER * cp)) - 1

def legal(fen):
    try:
        board = chess.Board(fen)
        return board.is_valid()
    except ValueError:
        return False

# def unique(stockfish: Stockfish, fen):
#     stockfish.send_ucinewgame_command()
#     stockfish.set_fen_position(fen)
#     return _unique(stockfish)

# def _unique(stockfish: Stockfish):
#     TAU_UNI = 0.5

#     moves = stockfish.get_top_moves(2)

#     if len(moves) == 0: return False  # if you have no moves, not a unique solution
#     # if len(moves) == 1: return moves[0]["Centipawn"] > 0 if moves[0]["Mate"] is None else moves[0]["Mate"] > 0  # if only one move, it must be a good one (NOTE: not the same as the paper)
#     if len(moves) == 1: return True

#     best_move = moves[0]
#     second_move = moves[1]
#     if best_move["Mate"] is None:
#         if second_move["Mate"] is not None: return True  # if second move is opponent mate, winning
#         return (calculate_winning_chances(best_move["Centipawn"]) - calculate_winning_chances(second_move["Centipawn"])) >= TAU_UNI
    
#     # here best move is checkmate

#     if best_move["Mate"] < 0: return False  # opponent mate is not puzzle (all moves bad)
#     if second_move["Mate"] is not None and second_move["Mate"] > 0: return False  # two moves lead to mate is not a puzzle (2 or more moves are good)

#     stockfish.make_moves_from_current_position([best_move["Move"]])

#     opponent_response = stockfish.get_best_move()
#     if opponent_response is None:
#         return True

#     stockfish.make_moves_from_current_position([opponent_response])
#     return _unique(stockfish)

def unique(fen, engine_path, depth=15):
    board = chess.Board(fen)
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        return _unique(board, engine, depth)

def _unique(board: chess.Board, engine: chess.engine.SimpleEngine, depth):
    TAU_UNI = 0.5

    info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=2)

    if len(info) == 0: return False
    if len(info) == 1: return True

    best_info = info[0]
    second_info = info[1]

    best_score = best_info["score"].pov(board.turn)
    second_score = second_info["score"].pov(board.turn)

    if not best_score.is_mate():
        if second_score.is_mate():  # enemy checkmates you (only 1 good move that being best_move)
            return True
        return (calculate_winning_chances(best_score.score()) - calculate_winning_chances(second_score.score())) >= TAU_UNI

    if best_score.mate() < 0: return False
    if second_score.is_mate() and second_score.mate() > 0: return False

    best_move = best_info["pv"][0]
    board.push(best_move)

    if board.is_game_over():
        return True

    opp_info = engine.analyse(board, chess.engine.Limit(depth=depth))
    
    if "pv" not in opp_info:
        return True
        
    opponent_response = opp_info["pv"][0]
    board.push(opponent_response)

    if board.is_game_over():
        return True

    return _unique(board, engine, depth)

# def counter_intuitive(stockfish: Stockfish, fen: str) -> bool:
#     stockfish.set_fen_position(fen)
    
#     ground_truth_depth = stockfish.get_depth()
#     ground_truth_move = stockfish.get_best_move()
#     if not ground_truth_move:
#         return False

#     found_at_depth = stockfish.get_depth()
#     stockfish.send_ucinewgame_command()
#     for d in range(1, ground_truth_depth + 1):
#         stockfish.set_depth(d)
#         stockfish.set_fen_position(fen)
#         move_at_depth = stockfish.get_best_move()
#         if move_at_depth == ground_truth_move:
#             found_at_depth = d
#             break
#     v_critical_point = found_at_depth - 1
#     stockfish.set_depth(ground_truth_depth)

#     board = chess.Board(fen)
#     move_obj = chess.Move.from_uci(ground_truth_move)
    
#     captured_val = 0
#     if board.is_capture(move_obj):
#         if board.is_en_passant(move_obj):
#             captured_val = 1
#         else:
#             target_piece = board.piece_at(move_obj.to_square)
#             if target_piece:
#                 piece_values = {
#                     chess.PAWN: 1, 
#                     chess.KNIGHT: 3, 
#                     chess.BISHOP: 3, 
#                     chess.ROOK: 5, 
#                     chess.QUEEN: 9,
#                     chess.KING: 0
#                 }
#                 captured_val = piece_values.get(target_piece.piece_type, 0)
    
#     v_capture_material = - (captured_val / 9.0)
#     r_cnt = 0.8 * v_critical_point + 0.1 * v_capture_material
#     return r_cnt > 0.1

def counter_intuitive(fen, engine_path, depth=15):
    board = chess.Board(fen)
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        history = []        
        with engine.analysis(board, chess.engine.Limit(depth=depth)) as analysis:
            for info in analysis:
                if "pv" in info and "depth" in info:
                    move_depth = info["depth"]
                    best_move = info["pv"][0]
                    history.append((move_depth, best_move))

        critical_point = depth
        
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
        return score > 0.1


if __name__ == "__main__":
    stockfish_path = "./Stockfish/src/stockfish"

    assert unique("r5k1/1b3NPp/p3r1n1/2pq4/3p4/8/PPP2QPP/4RRK1 w - - 0 1", stockfish_path)  # one good move by evaluation
    assert unique("rq2rn1k/p1p1n1R1/bp5P/4qBp1/1P1p3N/PQP5/3K1P1P/7R w - - 0 1", stockfish_path)  # checkmate
    assert unique("7r/4R1R1/2r2p1p/p5pk/2p5/P1P3PP/1n5K/8 w - - 0 1", stockfish_path)  # checkmate
    assert not unique("8/1KP5/8/2p5/1pP5/p7/k7/1R3R2 w - - 0 1", stockfish_path)  # many options for checkmate
    assert unique("r4b1k/pq1PN1pp/nn1Q4/2p3P1/2p4P/1Pp5/P7/2KR4 w - - 0 1", stockfish_path)  # one good move by evaluation
    assert unique("1q4rk/ppr1PQpp/1b3R2/3R4/1P6/4P3/P5PP/6K1 w - - 0 1", stockfish_path, depth=24)  # one good move by evaluation (depth 24 finds it)
    assert unique("rr4k1/1R3p2/4pPp1/3pPP1p/3qn1PK/8/6B1/2Q2R2 w - - 0 1", stockfish_path, depth=30)  # one good move by evaluation (depth 30 finds it)

    assert counter_intuitive("r5k1/1b3NPp/p3r1n1/2pq4/3p4/8/PPP2QPP/4RRK1 w - - 0 1", stockfish_path)
    assert counter_intuitive("rq2rn1k/p1p1n1R1/bp5P/4qBp1/1P1p3N/PQP5/3K1P1P/7R w - - 0 1", stockfish_path)
    assert counter_intuitive("7r/4R1R1/2r2p1p/p5pk/2p5/P1P3PP/1n5K/8 w - - 0 1", stockfish_path)
    assert counter_intuitive("8/1KP5/8/2p5/1pP5/p7/k7/1R3R2 w - - 0 1", stockfish_path)
    assert counter_intuitive("r4b1k/pq1PN1pp/nn1Q4/2p3P1/2p4P/1Pp5/P7/2KR4 w - - 0 1", stockfish_path)
    assert not counter_intuitive("rR4k1/5p2/4pPp1/3pPP1p/3qn1PK/8/6B1/2Q2R2 b - - 0 1", stockfish_path)
