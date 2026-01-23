from typing import List, Optional, Tuple
import math

import chess
from chess.engine import Score, SimpleEngine, Limit
from chess import square_rank, Color, Board, Square, Piece, square_distance
from chess import KING, QUEEN, ROOK, BISHOP, KNIGHT, PAWN
from chess.pgn import ChildNode, GameNode

from metrics.model import EngineMove, NextMovePair, Puzzle

# below from https://github.com/ornicar/lichess-puzzler/blob/master/tagger/util.py

def win_chances(score: Score):
    # from https://github.com/ornicar/lichess-puzzler/blob/master/tagger/zugzwang.py
    # from https://lichess.org/page/accuracy and https://arxiv.org/pdf/2402.04494 and other papers
    MULTIPLIER = 0.00368208
    mate = score.mate()
    if mate is not None:
        return 1 if mate > 0 else -1

    cp = score.score()
    return 2 / (1 + math.exp(-MULTIPLIER * cp)) - 1 if cp is not None else 0


def moved_piece_type(node: ChildNode) -> chess.PieceType:
    pt = node.board().piece_type_at(node.move.to_square)
    assert(pt)
    return pt

def is_advanced_pawn_move(node: ChildNode) -> bool:
    if node.move.promotion:
        return True
    if moved_piece_type(node) != chess.PAWN:
        return False
    to_rank = square_rank(node.move.to_square)
    return to_rank < 3 if node.turn() else to_rank > 4

def is_very_advanced_pawn_move(node: ChildNode) -> bool:
    if not is_advanced_pawn_move(node):
        return False
    to_rank = square_rank(node.move.to_square)
    return to_rank < 2 if node.turn() else to_rank > 5

def is_king_move(node: ChildNode) -> bool:
    return moved_piece_type(node) == chess.KING

def is_castling(node: ChildNode) -> bool:
    return is_king_move(node) and square_distance(node.move.from_square, node.move.to_square) > 1

def is_capture(node: ChildNode) -> bool:
    return node.parent.board().is_capture(node.move)

def next_node(node: ChildNode) -> Optional[ChildNode]:
    return node.variations[0] if node.variations else None

def next_next_node(node: ChildNode) -> Optional[ChildNode]:
    nn = next_node(node)
    return next_node(nn) if nn else None

values = { PAWN: 1, KNIGHT: 3, BISHOP: 3, ROOK: 5, QUEEN: 9 }
king_values = { PAWN: 1, KNIGHT: 3, BISHOP: 3, ROOK: 5, QUEEN: 9, KING: 99 }
ray_piece_types = [QUEEN, ROOK, BISHOP]

def piece_value(piece_type: chess.PieceType) -> int:
    return values[piece_type]

def material_count(board: Board, side: Color) -> int:
    return sum(len(board.pieces(piece_type, side)) * value for piece_type, value in values.items())

def material_diff(board: Board, side: Color) -> int:
    return material_count(board, side) - material_count(board, not side)

def attacked_opponent_pieces(board: Board, from_square: Square, pov: Color) -> List[Piece]:
    return [piece for (piece, _) in attacked_opponent_squares(board, from_square, pov)]

def attacked_opponent_squares(board: Board, from_square: Square, pov: Color) -> List[Tuple[Piece, Square]]:
    pieces = []
    for attacked_square in board.attacks(from_square):
        attacked_piece = board.piece_at(attacked_square)
        if attacked_piece and attacked_piece.color != pov:
            pieces.append((attacked_piece, attacked_square))
    return pieces

def is_defended(board: Board, piece: Piece, square: Square) -> bool:
    if board.attackers(piece.color, square):
        return True
    # ray defense https://lichess.org/editor/6k1/3q1pbp/2b1p1p1/1BPp4/rp1PnP2/4PRNP/4Q1P1/4B1K1_w_-_-_0_1
    for attacker in board.attackers(not piece.color, square):
        attacker_piece = board.piece_at(attacker)
        assert(attacker_piece)
        if attacker_piece.piece_type in ray_piece_types:
            bc = board.copy(stack = False)
            bc.remove_piece_at(attacker)
            if bc.attackers(piece.color, square):
                return True

    return False

def is_hanging(board: Board, piece: Piece, square: Square) -> bool:
    return not is_defended(board, piece, square)

def can_be_taken_by_lower_piece(board: Board, piece: Piece, square: Square) -> bool:
    for attacker_square in board.attackers(not piece.color, square):
        attacker = board.piece_at(attacker_square)
        assert(attacker)
        if attacker.piece_type != chess.KING and values[attacker.piece_type] < values[piece.piece_type]:
            return True
    return False

def is_in_bad_spot(board: Board, square: Square) -> bool:
    # hanging or takeable by lower piece
    piece = board.piece_at(square)
    assert(piece)
    return (bool(board.attackers(not piece.color, square)) and
            (is_hanging(board, piece, square) or can_be_taken_by_lower_piece(board, piece, square)))

def is_trapped(board: Board, square: Square) -> bool:
    if board.is_check() or board.is_pinned(board.turn, square):
        return False
    piece = board.piece_at(square)
    assert(piece)
    if piece.piece_type in [PAWN, KING]:
        return False
    if not is_in_bad_spot(board, square):
        return False
    for escape in board.legal_moves:
        if escape.from_square == square:
            capturing = board.piece_at(escape.to_square)
            if capturing and values[capturing.piece_type] >= values[piece.piece_type]:
                return False
            board.push(escape)
            if not is_in_bad_spot(board, escape.to_square):
                return False
            board.pop()
    return True

def attacker_pieces(board: Board, color: Color, square: Square) -> List[Piece]:
    return [p for p in [board.piece_at(s) for s in board.attackers(color, square)] if p]


# Below from https://github.com/ornicar/lichess-puzzler/blob/master/generator/util.py

def material_count(board: Board, side: Color) -> int:
    values = { chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9 }
    return sum(len(board.pieces(piece_type, side)) * value for piece_type, value in values.items())

def material_diff(board: Board, side: Color) -> int:
    return material_count(board, side) - material_count(board, not side)

def is_up_in_material(board: Board, side: Color) -> bool:
    return material_diff(board, side) > 0

def maximum_castling_rights(board: chess.Board) -> chess.Bitboard:
    return (
        (board.pieces_mask(chess.ROOK, chess.WHITE) & (chess.BB_A1 | chess.BB_H1) if board.king(chess.WHITE) == chess.E1 else chess.BB_EMPTY) |
        (board.pieces_mask(chess.ROOK, chess.BLACK) & (chess.BB_A8 | chess.BB_H8) if board.king(chess.BLACK) == chess.E8 else chess.BB_EMPTY)
    )


def get_next_move_pair(engine: SimpleEngine, node: GameNode, winner: Color, limit: Limit) -> NextMovePair:
    info = engine.analyse(node.board(), multipv = 2, limit = limit)
    best = EngineMove(info[0]["pv"][0], info[0]["score"].pov(winner))
    second = EngineMove(info[1]["pv"][0], info[1]["score"].pov(winner)) if len(info) > 1 else None
    return NextMovePair(node, winner, best, second)

def time_control_tier(line: str) -> Optional[int]:
    if not line.startswith("[TimeControl "):
        return None
    try:
        seconds, increment = line[1:][:-2].split()[1].replace("\"", "").split("+")
        total = int(seconds) + int(increment) * 40
        if total >= 480:
            return 3
        if total >= 180:
            return 2
        if total > 60:
            return 1
        return 0
    except:
        return 0
    
def count_mates(board:chess.Board) -> int:
    mates = 0
    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            mates += 1
        board.pop()
    return mates

def rating_tier(line: str) -> Optional[int]:
    if not line.startswith("[WhiteElo ") and not line.startswith("[BlackElo "):
        return None
    try:
        rating = int(line[11:15])
        if rating > 1750:
            return 3
        if rating > 1600:
            return 2
        if rating > 1500:
            return 1
        return 0
    except:
        return 0

# from lichess-puzzler commit 86744ff or https://github.com/lichess-org/scalachess/blob/master/core/src/main/scala/Divider.scala (slightly different, but mostly the same)

def _majors_and_minors(board: chess.Board) -> int:
    count = 0
    for piece in board.piece_map().values():
        if piece.piece_type != chess.PAWN and piece.piece_type != chess.KING:
            count += 1
    return count

def _backrank_sparse(board: chess.Board) -> bool:
    # Check White Backrank (Rank 1)
    white_backrank_count = 0
    for sq in range(0, 8):
        piece = board.piece_at(sq)
        if piece and piece.color == chess.WHITE:
            white_backrank_count += 1
    
    # Check Black Backrank (Rank 8)
    black_backrank_count = 0
    for sq in range(56, 64):
        piece = board.piece_at(sq)
        if piece and piece.color == chess.BLACK:
            black_backrank_count += 1

    return white_backrank_count < 4 or black_backrank_count < 4

def _score(white: int, black: int, y: int) -> int:
    counts = (white, black)

    if counts == (0, 0):
        return 0
    
    elif counts == (1, 0):
        return 1 + (8 - y)
    elif counts == (2, 0):
        return (2 + (y - 2)) if y > 2 else 0
    elif counts == (3, 0):
        return (3 + (y - 1)) if y > 1 else 0
    elif counts == (4, 0):
        return (3 + (y - 1)) if y > 1 else 0

    elif counts == (0, 1):
        return 1 + y
    elif counts == (1, 1):
        return 5 + abs(3 - y)
    elif counts == (2, 1):
        return 4 + y
    elif counts == (3, 1):
        return 5 + y

    elif counts == (0, 2):
        return (2 + (6 - y)) if y < 6 else 0
    elif counts == (1, 2):
        return 4 + (6 - y)
    elif counts == (2, 2):
        return 7
        
    elif counts == (0, 3):
        return (3 + (7 - y)) if y < 7 else 0
    elif counts == (1, 3):
        return 5 + (6 - y)

    elif counts == (0, 4):
        return (3 + (7 - y)) if y < 7 else 0

    return 0

def _generate_mixedness_regions():
    regions = []
    for rank in range(7):
        for file in range(7):
            sqs = [
                chess.square(file, rank),
                chess.square(file + 1, rank),
                chess.square(file, rank + 1),
                chess.square(file + 1, rank + 1)
            ]
            regions.append(sqs)
    return regions

mixedness_regions = _generate_mixedness_regions()

def _mixedness(board: chess.Board) -> int:
    total_score = 0
    
    for region in mixedness_regions:
        white_count = 0
        black_count = 0
        
        y = chess.square_rank(region[0]) + 1

        for sq in region:
            piece = board.piece_at(sq)
            if piece:
                if piece.color == chess.WHITE:
                    white_count += 1
                else:
                    black_count += 1
        
        total_score += _score(white_count, black_count, y)
        
    return total_score

def game_phase(puzzle: Puzzle) -> str:
    board = puzzle.game.board()
    m_and_m = _majors_and_minors(board)
    is_opening = not (
        m_and_m <= 10 or 
        _backrank_sparse(board) or 
        _mixedness(board) > 150
    )

    is_endgame = (not is_opening) and (m_and_m <= 6)

    if is_opening:
        return "opening"
    elif is_endgame:
        return "endgame"
    else:
        return "middlegame"

