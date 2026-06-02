import chess

from .diversity_filtering import board_distance, PV_distance, get_board_distance, get_pv_distance


piece_counts = {chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2, chess.ROOK: 2, chess.QUEEN: 1, chess.KING: 1}
pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
colors = [chess.WHITE, chess.BLACK]

def good_piece_counts(fen):
    board = chess.Board(fen)
    for color in colors:
        for piece in pieces:
            if len(board.pieces(piece, color)) > piece_counts[piece]:
                return False
    return True

def inter_batch_distances(fen, pv, sampled_fens, sampled_pvs):
    min_board_dist = float('inf')
    best_pv = None
    for sampled_fen, sampled_pv in zip(sampled_fens, sampled_pvs):
        bd = get_board_distance(fen, sampled_fen)
        if bd < min_board_dist:
            min_board_dist = bd
            best_pv = sampled_pv
    min_pv_dist = get_pv_distance(pv, best_pv) if (pv and best_pv) else 0
    return min_board_dist if min_board_dist != float('inf') else 0, min_pv_dist

def intra_batch_distances(fen, pv, fens, pvs, i):
    min_board_dist = float('inf')
    best_pv = None
    for index, (other_fen, other_pv) in enumerate(zip(fens, pvs)):
        if other_fen is None or index == i:
            continue
        bd = get_board_distance(fen, other_fen)
        if bd < min_board_dist:
            min_board_dist = bd
            best_pv = other_pv
    min_pv_dist = get_pv_distance(pv, best_pv) if (pv and best_pv) else 0
    return min_board_dist if min_board_dist != float('inf') else 0, min_pv_dist
