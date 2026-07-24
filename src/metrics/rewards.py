import chess

from .diversity_filtering import board_distance, PV_distance, get_board_distance, get_pv_distance, get_opponent_pv_distance, get_abstracted_pv, get_abstracted_pv_hamming_distance


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
    for index, other_fen in enumerate(fens):
        if other_fen is None or index == i:
            continue
        bd = get_board_distance(fen, other_fen)
        if bd < min_board_dist:
            min_board_dist = bd

    min_pv_dist = float('inf')
    if pv:
        for index, other_pv in enumerate(pvs):
            if other_pv is None or index == i:
                continue
            if other_pv:
                pd = get_pv_distance(pv, other_pv)
                if pd < min_pv_dist:
                    min_pv_dist = pd
    
    min_opponent_pv_dist = float('inf')
    if pv:
        for index, other_pv in enumerate(pvs):
            if other_pv is None or index == i:
                continue
            if other_pv:
                opd = get_opponent_pv_distance(pv, other_pv)
                if opd < min_opponent_pv_dist:
                    min_opponent_pv_dist = opd

    min_abstracted_pv_dist = float('inf')
    if pv:
        apv = get_abstracted_pv(fen, pv)
        for index, other_pv in enumerate(pvs):
            if other_pv is None or index == i:
                continue
            other_fen = fens[index]
            if other_fen and other_pv:
                other_apv = get_abstracted_pv(other_fen, other_pv)
                apd = get_abstracted_pv_hamming_distance(apv, other_apv)
                if apd < min_abstracted_pv_dist:
                    min_abstracted_pv_dist = apd

    return (
        min_board_dist if min_board_dist != float('inf') else 0,
        min_pv_dist if min_pv_dist != float('inf') else 0,
        min_opponent_pv_dist if min_opponent_pv_dist != float('inf') else 0,
        min_abstracted_pv_dist if min_abstracted_pv_dist != float('inf') else 0
    )
