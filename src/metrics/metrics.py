# import chess
# from chess.engine import SimpleEngine, Limit, Score

# from time import perf_counter

# from util import win_chances


# def legal(fen):
#     try:
#         board = chess.Board(fen)
#         return board.is_valid()
#     except ValueError:
#         return False

# def unique(fen, engine: SimpleEngine, depth=15):
#     board = chess.Board(fen)
#     return _unique(board, engine, depth)

# def _unique(board: chess.Board, engine: SimpleEngine, depth):
#     TAU_UNI = 0.5

#     info = engine.analyse(board, Limit(depth=depth), multipv=2)

#     if len(info) == 0: return False
#     if len(info) == 1: return True

#     best_info = info[0]
#     second_info = info[1]

#     best_score = best_info["score"].pov(board.turn)
#     second_score = second_info["score"].pov(board.turn)

#     if not best_score.is_mate():
#         return (win_chances(best_score) - win_chances(second_score)) >= TAU_UNI

#     if best_score.mate() < 0: return False
#     if second_score.is_mate() and second_score.mate() > 0: return False

#     best_move = best_info["pv"][0]
#     board.push(best_move)

#     if board.is_game_over():
#         return True

#     opp_info = engine.analyse(board, Limit(depth=depth))

#     if "pv" not in opp_info:
#         return True

#     opponent_response = opp_info["pv"][0]
#     board.push(opponent_response)

#     if board.is_game_over():
#         return True

#     return _unique(board, engine, depth)

# def counter_intuitive(fen, engine: SimpleEngine, depth=15):
#     board = chess.Board(fen)
#     history = []        
#     with engine.analysis(board, Limit(depth=depth)) as analysis:
#         for info in analysis:
#             if "pv" in info and "depth" in info:
#                 move_depth = info["depth"]
#                 best_move = info["pv"][0]
#                 history.append((move_depth, best_move))

#     critical_point = depth
    
#     final_best_move = history[-1][1]
#     for move_depth, move in history:
#         if move == final_best_move:
#             critical_point = move_depth
#             break
    
#     v_critical_point = critical_point - 1

#     v_capture_material = 0
#     if board.is_capture(final_best_move):
#         if board.is_en_passant(final_best_move):
#             captured_value = 1
#         else:
#             captured_piece = board.piece_at(final_best_move.to_square)
#             values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
#             captured_value = values.get(captured_piece.piece_type, 0)
        
#         v_capture_material = -captured_value / 9

#     score = (0.8 * v_critical_point) + (0.1 * v_capture_material)
#     return score > 0.1


# if __name__ == "__main__":
#     stockfish_path = "./Stockfish/src/stockfish"
#     engine = SimpleEngine.popen_uci(stockfish_path)

#     start = perf_counter()
#     assert unique("r5k1/1b3NPp/p3r1n1/2pq4/3p4/8/PPP2QPP/4RRK1 w - - 0 1", engine)  # one good move by evaluation
#     assert unique("rq2rn1k/p1p1n1R1/bp5P/4qBp1/1P1p3N/PQP5/3K1P1P/7R w - - 0 1", engine)  # checkmate
#     assert unique("7r/4R1R1/2r2p1p/p5pk/2p5/P1P3PP/1n5K/8 w - - 0 1", engine)  # checkmate
#     assert not unique("8/1KP5/8/2p5/1pP5/p7/k7/1R3R2 w - - 0 1", engine)  # many options for checkmate
#     assert unique("r4b1k/pq1PN1pp/nn1Q4/2p3P1/2p4P/1Pp5/P7/2KR4 w - - 0 1", engine)  # one good move by evaluation
#     assert unique("1q4rk/ppr1PQpp/1b3R2/3R4/1P6/4P3/P5PP/6K1 w - - 0 1", engine, depth=25)  # one good move by evaluation (depth 25 finds it)
#     assert unique("rr4k1/1R3p2/4pPp1/3pPP1p/3qn1PK/8/6B1/2Q2R2 w - - 0 1", engine, depth=30)  # one good move by evaluation (depth 30 finds it)

#     assert counter_intuitive("r5k1/1b3NPp/p3r1n1/2pq4/3p4/8/PPP2QPP/4RRK1 w - - 0 1", engine)
#     assert counter_intuitive("rq2rn1k/p1p1n1R1/bp5P/4qBp1/1P1p3N/PQP5/3K1P1P/7R w - - 0 1", engine)
#     assert counter_intuitive("7r/4R1R1/2r2p1p/p5pk/2p5/P1P3PP/1n5K/8 w - - 0 1", engine)
#     assert counter_intuitive("8/1KP5/8/2p5/1pP5/p7/k7/1R3R2 w - - 0 1", engine)
#     assert counter_intuitive("r4b1k/pq1PN1pp/nn1Q4/2p3P1/2p4P/1Pp5/P7/2KR4 w - - 0 1", engine)
#     assert not counter_intuitive("rR4k1/5p2/4pPp1/3pPP1p/3qn1PK/8/6B1/2Q2R2 b - - 0 1", engine)

#     print(perf_counter() - start)

#     engine.quit()
