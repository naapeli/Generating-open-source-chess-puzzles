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
    for sampled_fen, sampled_pv in zip(sampled_fens, sampled_pvs):
        if not board_distance(fen, sampled_fen):
            return False, True
        if not PV_distance(sampled_pv, pv):
            return True, False
    return True, True

def good_intra_batch_distances(fen, pv, puzzles):
    for other_puzzle in puzzles:
        if other_puzzle is None:
            continue
        other_fen = other_puzzle.game.board().fen()
        other_pv = " ".join([move.uci() for move in other_puzzle.mainline])
        if not board_distance(fen, other_fen):
            return False, True
        if not PV_distance(pv, other_pv):
            return True, False
    return True, True
