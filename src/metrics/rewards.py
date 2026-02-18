import chess

from .diversity_filtering import board_distance, PV_distance


piece_counts = {chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2, chess.ROOK: 2, chess.QUEEN: 1, chess.KING: 1}
pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
colors = [chess.WHITE, chess.BLACK]

def good_piece_counts(puzzle):
    for color in colors:
        for piece in pieces:
            if len(puzzle.game.board().pieces(piece, color)) > piece_counts[piece]:
                return False
    return True

def good_inter_batch_distances(fen, pv, sampled_fens, sampled_pvs):
    good_board_distance = True
    good_pv_distance = True
    for sampled_fen, sampled_pv in zip(sampled_fens, sampled_pvs):
        good_board_distance = good_board_distance and board_distance(fen, sampled_fen)
        good_pv_distance = good_pv_distance and PV_distance(sampled_pv, pv)
        if not good_board_distance and not good_pv_distance:  # if both are bad
            return good_board_distance, good_pv_distance
    return good_board_distance, good_pv_distance

def good_intra_batch_distances(fen, pv, puzzles, i):
    good_board_distance = True
    good_pv_distance = True
    for index, other_puzzle in enumerate(puzzles):
        if other_puzzle is None or index == i:
            continue
        other_fen = other_puzzle.game.board().fen()
        other_pv = " ".join([move.uci() for move in other_puzzle.mainline])
        good_board_distance = good_board_distance and board_distance(fen, other_fen)
        good_pv_distance = good_pv_distance and PV_distance(pv, other_pv)
        if not good_board_distance and not good_pv_distance:  # if both are bad
            return good_board_distance, good_pv_distance
    return good_board_distance, good_pv_distance
