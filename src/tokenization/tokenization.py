import re
from enum import IntEnum, auto

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


# =========================================== FENS ===========================================

class FENTokens(IntEnum):
    # 48 tokens
    no_piece = 0
    white_pawn = auto()
    white_knight = auto()
    white_bishop = auto()
    white_rook = auto()
    white_queen = auto()
    white_king = auto()
    black_pawn = auto()
    black_knight = auto()
    black_bishop = auto()
    black_rook = auto()
    black_queen = auto()
    black_king = auto()

    side_white = auto()
    side_black = auto()

    no_castle = auto()
    castle_white_king = auto()
    castle_white_queen = auto()
    castle_black_king = auto()
    castle_black_queen = auto()
    
    no_en_passant = auto()
    en_passant_1 = auto()
    en_passant_2 = auto()
    en_passant_3 = auto()
    en_passant_4 = auto()
    en_passant_5 = auto()
    en_passant_6 = auto()
    en_passant_7 = auto()
    en_passant_8 = auto()
    en_passant_a = auto()
    en_passant_b = auto()
    en_passant_c = auto()
    en_passant_d = auto()
    en_passant_e = auto()
    en_passant_f = auto()
    en_passant_g = auto()
    en_passant_h = auto()

    pad_counter = auto()
    counter_0 = auto()
    counter_1 = auto()
    counter_2 = auto()
    counter_3 = auto()
    counter_4 = auto()
    counter_5 = auto()
    counter_6 = auto()
    counter_7 = auto()
    counter_8 = auto()
    counter_9 = auto()

    mask = auto()


board_tokens = [
    (".", FENTokens.no_piece), ("P", FENTokens.white_pawn), ("N", FENTokens.white_knight),
    ("B", FENTokens.white_bishop), ("R", FENTokens.white_rook), ("Q", FENTokens.white_queen),
    ("K", FENTokens.white_king), ("p", FENTokens.black_pawn), ("n", FENTokens.black_knight),
    ("b", FENTokens.black_bishop), ("r", FENTokens.black_rook), ("q", FENTokens.black_queen),
    ("k", FENTokens.black_king)
]
board_str_2_token = dict(board_tokens)
board_token_2_str = dict([(token, string) for string, token in board_tokens])

side_tokens = [
    ("w", FENTokens.side_white), ("b", FENTokens.side_black)
]
side_str_2_token = dict(side_tokens)
side_token_2_str = dict([(token, string) for string, token in side_tokens])

castling_tokens = [
    ("-", FENTokens.no_castle), ("K", FENTokens.castle_white_king), ("Q", FENTokens.castle_white_queen),
    ("k", FENTokens.castle_black_king), ("q", FENTokens.castle_black_queen)
]
castling_str_2_token = dict(castling_tokens)
castling_token_2_str = dict([(token, string) for string, token in castling_tokens])

enpassant_tokens = [
    ("-", FENTokens.no_en_passant), ("1", FENTokens.en_passant_1), ("2", FENTokens.en_passant_2),
    ("3", FENTokens.en_passant_3), ("4", FENTokens.en_passant_4), ("5", FENTokens.en_passant_5),
    ("6", FENTokens.en_passant_6), ("7", FENTokens.en_passant_7), ("8", FENTokens.en_passant_8),
    ("a", FENTokens.en_passant_a), ("b", FENTokens.en_passant_b), ("c", FENTokens.en_passant_c),
    ("d", FENTokens.en_passant_d), ("e", FENTokens.en_passant_e), ("f", FENTokens.en_passant_f),
    ("g", FENTokens.en_passant_g), ("h", FENTokens.en_passant_h),
]
enpassant_str_2_token = dict(enpassant_tokens)
enpassant_token_2_str = dict([(token, string) for string, token in enpassant_tokens])

counter_tokens = [
    (".", FENTokens.pad_counter), ("1", FENTokens.counter_1), ("2", FENTokens.counter_2),
    ("3", FENTokens.counter_3), ("4", FENTokens.counter_4), ("5", FENTokens.counter_5),
    ("6", FENTokens.counter_6), ("7", FENTokens.counter_7), ("8", FENTokens.counter_8),
    ("9", FENTokens.counter_9), ("0", FENTokens.counter_0)
]
counter_str_2_token = dict(counter_tokens)
counter_token_2_str = dict([(token, string) for string, token in counter_tokens])

# def make_first_move()

def tokenize_fen(fen):
    board, side, castling, enpassant, halfmove, fullmove = fen.split(" ")
    board = re.sub(r"\d", lambda digit: "." * int(digit.group()), board)
    board = re.sub("/", "", board)
    board = [board_str_2_token[char] for char in board]

    castling_characters = "KQkq"
    castling_tokens = [FENTokens.castle_white_king, FENTokens.castle_white_queen, FENTokens.castle_black_king, FENTokens.castle_black_queen]
    castling = [token if char in castling else FENTokens.no_castle for token, char in zip(castling_tokens, castling_characters)]
    
    enpassant = [enpassant_str_2_token[enpassant[0]], enpassant_str_2_token[enpassant[1]]] if enpassant != "-" else [FENTokens.no_en_passant, FENTokens.no_en_passant]

    halfmove = "." + halfmove if len(halfmove) == 1 else halfmove
    halfmove = [counter_str_2_token[halfmove[0]], counter_str_2_token[halfmove[1]]]
    assert len(halfmove) == 2, "halfmove counter is encoded as a number with 2 digits"
    fullmove = ".." + fullmove if len(fullmove) == 1 else "." + fullmove if len(fullmove) == 2 else fullmove
    fullmove = [counter_str_2_token[fullmove[0]], counter_str_2_token[fullmove[1]], counter_str_2_token[fullmove[2]]]
    assert len(fullmove) == 3, "fullmove counter is encoded as a number with 3 digits"

    return board + [side_str_2_token[side]] + castling + enpassant + halfmove + fullmove


# =========================================== THEMES ===========================================

unique_themes = ['crushing', 'hangingPiece', 'long', 'middlegame', 'advantage', 'endgame', 'short', 'rookEndgame', 'fork', 'pawnEndgame', 'mate', 'mateIn2', 'master', 'interference', 'kingsideAttack', 'veryLong', 'zugzwang', 'exposedKing', 'skewer', 'mateIn1', 'oneMove', 'opening', 'pin', 'quietMove', 'backRankMate', 'discoveredAttack', 'sacrifice', 'bishopEndgame', 'bodenMate', 'deflection', 'smotheredMate', 'advancedPawn', 'attraction', 'promotion', 'mateIn3', 'masterVsMaster', 'superGM', 'queensideAttack', 'knightEndgame', 'cornerMate', 'defensiveMove', 'queenEndgame', 'attackingF2F7', 'queenRookEndgame', 'clearance', 'intermezzo', 'equality', 'trappedPiece', 'hookMate', 'xRayAttack', 'capturingDefender', 'doubleBishopMate', 'doubleCheck', 'arabianMate', 'mateIn4', 'enPassant', 'vukovicMate', 'dovetailMate', 'triangleMate', 'balestraMate', 'killBoxMate', 'anastasiaMate', 'blindSwineMate', 'castling', 'mateIn5', 'underPromotion']

theme_preprocessor = MultiLabelBinarizer(classes = unique_themes)
theme_preprocessor.fit([])

def tokenize_themes(themes):
    tokens = theme_preprocessor.transform(themes.str.split())
    return tokens


# =========================================== RATINGS ===========================================

MIN_RATING = 399
MAX_RATING = 3395

def scale_ratings(ratings):
    return (ratings - MIN_RATING) / (MAX_RATING - MIN_RATING)


# =========================================== TOKENIZE ===========================================

def tokenize(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fens = np.stack(df["Puzzle_FEN"].apply(tokenize_fen).values, dtype=np.int8)
    themes = tokenize_themes(df["Themes"]).astype(np.int8)
    ratings = scale_ratings(df["Rating"].astype(np.float16)).to_numpy()
    return torch.from_numpy(fens), torch.from_numpy(themes), torch.from_numpy(ratings)

def tokens_to_fen(tokens: torch.Tensor) -> str:
    try:
        tokens = tokens.squeeze()
        if tokens.ndim != 1 and tokens.size(0) != 76:
            raise RuntimeError()
        
        tokens = tokens.tolist()
        board = "".join(board_token_2_str[token] for token in tokens[:64])
        board = "/".join(board[i:i + 8] for i in range(0, len(board), 8))
        board = re.sub(r"\.+", lambda string: str(len(string.group())), board)
        side = side_token_2_str[tokens[64]]
        castling = "".join(castling_token_2_str[token] for token in tokens[65:69])
        castling = "-" if castling == "----" else re.sub("-", "", castling)
        en_passant = "".join([enpassant_token_2_str[token] for token in tokens[69:71]])
        en_passant = "-" if en_passant == "--" else en_passant  # maybe check that one is not - and the other something else in the future (if the generative model generates an illegal puzzle)
        halfmove_counter = re.sub(r"\.", "", "".join(counter_token_2_str[token] for token in tokens[71:73]))
        fullmove_counter = re.sub(r"\.", "", "".join(counter_token_2_str[token] for token in tokens[73:76]))

        return " ".join([board, side, castling, en_passant, halfmove_counter, fullmove_counter])
    except KeyError:
        raise NotLegalPositionError("The tokens do not yield a legal position")


class NotLegalPositionError(Exception):
    pass


# def pad_fen(fen: str) -> str:
#     board, side, castling, enpassant, halfmove, fullmove = fen.split(" ")
#     board = re.sub(r"\d", lambda digit: "." * int(digit.group()), board)
#     board = re.sub("/", lambda _: "", board)  # remove the slashes

#     castling_characters = "KQkq"
#     castling = "".join([char if char in castling else "." for char in castling_characters])
    
#     enpassant = enpassant if enpassant != "-" else "-."

#     halfmove = "." + halfmove if len(halfmove) == 1 else halfmove
#     assert len(halfmove) == 2, "halfmove counter is encoded as a number with 2 digits"
#     fullmove = ".." + fullmove if len(fullmove) == 1 else "." + fullmove if len(fullmove) == 2 else fullmove
#     assert len(fullmove) == 3, "fullmove counter is encoded as a number with 3 digits"

#     return "".join([board, side, castling, enpassant, halfmove, fullmove])
