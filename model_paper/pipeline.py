import re
import torch
import torch.nn.functional as F
from enum import IntEnum, auto
from diffusers import DiffusionPipeline

# ==========================================
# 1. Custom Masking Schedules
# ==========================================

class MaskingSchedule:
    def __init__(self, eps=1e-4):
        self.eps = eps
    def alpha(self, t): raise NotImplementedError()
    def __call__(self, t): return self.alpha(t)

class LinearSchedule(MaskingSchedule):
    def alpha(self, t):
        val = 1.0 - t
        return (1.0 - 2 * self.eps) * val + self.eps

class CosineSchedule(MaskingSchedule):
    def alpha(self, t):
        val = 1.0 - torch.cos(torch.pi / 2.0 * (1.0 - t))
        return (1.0 - 2 * self.eps) * val + self.eps

class GeometricSchedule(MaskingSchedule):
    def __init__(self, beta_min=0.001, beta_max=1.0, **kwargs):
        super().__init__(**kwargs)
        self.beta_min = beta_min
        self.beta_max = beta_max
    def alpha(self, t):
        inner = self.beta_min ** (1.0 - t) * self.beta_max ** t
        val = torch.exp(-inner)
        return (1.0 - 2 * self.eps) * val + self.eps

class PolynomialSchedule(MaskingSchedule):
    def __init__(self, exponent=2.0, **kwargs):
        super().__init__(**kwargs)
        self.exponent = exponent
    def alpha(self, t):
        val = 1.0 - t ** self.exponent
        return (1.0 - 2 * self.eps) * val + self.eps

# ==========================================
# 2. Chess Tokenization Utilities
# ==========================================

class FENTokens(IntEnum):
    no_piece = 0
    white_pawn = auto(); white_knight = auto(); white_bishop = auto()
    white_rook = auto(); white_queen = auto(); white_king = auto()
    black_pawn = auto(); black_knight = auto(); black_bishop = auto()
    black_rook = auto(); black_queen = auto(); black_king = auto()
    side_white = auto(); side_black = auto()
    no_castle = auto()
    castle_white_king = auto(); castle_white_queen = auto()
    castle_black_king = auto(); castle_black_queen = auto()
    none = auto()
    rank1 = auto(); rank2 = auto(); rank3 = auto(); rank4 = auto()
    rank5 = auto(); rank6 = auto(); rank7 = auto(); rank8 = auto()
    filea = auto(); fileb = auto(); filec = auto(); filed = auto()
    filee = auto(); filef = auto(); fileg = auto(); fileh = auto()
    pad_counter = auto()
    counter_0 = auto(); counter_1 = auto(); counter_2 = auto()
    counter_3 = auto(); counter_4 = auto(); counter_5 = auto()
    counter_6 = auto(); counter_7 = auto(); counter_8 = auto()
    counter_9 = auto()
    promote_q = auto(); promote_r = auto(); promote_b = auto(); promote_n = auto()
    mask = auto()

board_token_mapping = [
    (".", FENTokens.no_piece), ("P", FENTokens.white_pawn), ("N", FENTokens.white_knight),
    ("B", FENTokens.white_bishop), ("R", FENTokens.white_rook), ("Q", FENTokens.white_queen),
    ("K", FENTokens.white_king), ("p", FENTokens.black_pawn), ("n", FENTokens.black_knight),
    ("b", FENTokens.black_bishop), ("r", FENTokens.black_rook), ("q", FENTokens.black_queen),
    ("k", FENTokens.black_king)
]
board_str_2_token = {char: token for char, token in board_token_mapping}
board_token_2_str = {token: char for char, token in board_token_mapping}

side_token_mapping = [("w", FENTokens.side_white), ("b", FENTokens.side_black)]
side_str_2_token = {char: token for char, token in side_token_mapping}
side_token_2_str = {token: char for char, token in side_token_mapping}

castling_token_mapping = [
    ("-", FENTokens.no_castle), ("K", FENTokens.castle_white_king), ("Q", FENTokens.castle_white_queen),
    ("k", FENTokens.castle_black_king), ("q", FENTokens.castle_black_queen)
]
castling_token_2_str = {token: char for char, token in castling_token_mapping}

enpassant_token_mapping = [
    ("-", FENTokens.none), ("1", FENTokens.rank1), ("2", FENTokens.rank2),
    ("3", FENTokens.rank3), ("4", FENTokens.rank4), ("5", FENTokens.rank5),
    ("6", FENTokens.rank6), ("7", FENTokens.rank7), ("8", FENTokens.rank8),
    ("a", FENTokens.filea), ("b", FENTokens.fileb), ("c", FENTokens.filec),
    ("d", FENTokens.filed), ("e", FENTokens.filee), ("f", FENTokens.filef),
    ("g", FENTokens.fileg), ("h", FENTokens.fileh)
]
enpassant_str_2_token = {char: token for char, token in enpassant_token_mapping}
enpassant_token_2_str = {token: char for char, token in enpassant_token_mapping}

counter_token_mapping = [
    (".", FENTokens.pad_counter), ("1", FENTokens.counter_1), ("2", FENTokens.counter_2),
    ("3", FENTokens.counter_3), ("4", FENTokens.counter_4), ("5", FENTokens.counter_5),
    ("6", FENTokens.counter_6), ("7", FENTokens.counter_7), ("8", FENTokens.counter_8),
    ("9", FENTokens.counter_9), ("0", FENTokens.counter_0)
]
counter_str_2_token = {char: token for char, token in counter_token_mapping}
counter_token_2_str = {token: char for char, token in counter_token_mapping}

promote_token_mapping = [
    ("q", FENTokens.promote_q), ("r", FENTokens.promote_r), 
    ("b", FENTokens.promote_b), ("n", FENTokens.promote_n), 
    ("-", FENTokens.none)
]
promote_str_2_token = {char: token for char, token in promote_token_mapping}
promote_token_2_str = {token: char for char, token in promote_token_mapping}

unique_themes = [
    'crushing', 'hangingPiece', 'long', 'middlegame', 'advantage', 'endgame', 'short', 'rookEndgame', 'fork',
    'pawnEndgame', 'mate', 'mateIn2', 'master', 'interference', 'kingsideAttack', 'veryLong', 'zugzwang',
    'exposedKing', 'skewer', 'mateIn1', 'oneMove', 'opening', 'pin', 'quietMove', 'backRankMate',
    'discoveredAttack', 'sacrifice', 'bishopEndgame', 'bodenMate', 'deflection', 'smotheredMate',
    'advancedPawn', 'attraction', 'promotion', 'mateIn3', 'masterVsMaster', 'superGM', 'queensideAttack',
    'knightEndgame', 'cornerMate', 'defensiveMove', 'queenEndgame', 'attackingF2F7', 'queenRookEndgame',
    'clearance', 'intermezzo', 'equality', 'trappedPiece', 'hookMate', 'xRayAttack', 'capturingDefender',
    'doubleBishopMate', 'doubleCheck', 'arabianMate', 'mateIn4', 'enPassant', 'vukovicMate', 'dovetailMate',
    'triangleMate', 'balestraMate', 'killBoxMate', 'anastasiaMate', 'blindSwineMate', 'castling',
    'mateIn5', 'underPromotion'
]
theme_to_idx = {theme: idx for idx, theme in enumerate(unique_themes)}

MIN_RATING = 399
MAX_RATING = 3395

def scale_rating(rating):
    return (float(rating) - MIN_RATING) / (MAX_RATING - MIN_RATING)

def tokens_to_fen(tokens_list):
    board = "".join(board_token_2_str[token] for token in tokens_list[:64])
    board = "/".join(board[i:i + 8] for i in range(0, len(board), 8))
    board = re.sub(r"\.+", lambda string: str(len(string.group())), board)
    
    side = side_token_2_str[tokens_list[64]]
    
    castling = "".join(castling_token_2_str[token] for token in tokens_list[65:69])
    castling = "-" if castling == "----" else re.sub("-", "", castling)
    
    en_passant = "".join([enpassant_token_2_str[token] for token in tokens_list[69:71]])
    en_passant = "-" if en_passant == "--" else en_passant
    
    halfmove_counter = re.sub(r"\.", "", "".join(counter_token_2_str[token] for token in tokens_list[71:73]))
    fullmove_counter = re.sub(r"\.", "", "".join(counter_token_2_str[token] for token in tokens_list[73:76]))
    
    return " ".join([board, side, castling, en_passant, halfmove_counter, fullmove_counter])

def tokens_to_move(tokens_list):
    move_tokens = tokens_list[-5:]
    m1 = enpassant_token_2_str.get(move_tokens[0], "")
    m2 = enpassant_token_2_str.get(move_tokens[1], "")
    m3 = enpassant_token_2_str.get(move_tokens[2], "")
    m4 = enpassant_token_2_str.get(move_tokens[3], "")
    promotion = promote_token_2_str.get(move_tokens[4], "")
    return m1 + m2 + m3 + m4 + (promotion if promotion != "-" else "")

def partial_fen_to_tokens(partial_fen: str, config, mask_token: int, best_move: str = None) -> torch.Tensor:
    parts = partial_fen.strip().split(" ")
    board_part = parts[0]
    side_part = parts[1] if len(parts) > 1 else "w"
    castling_part = parts[2] if len(parts) > 2 else "KQkq"
    ep_part = parts[3] if len(parts) > 3 else "-"
    halfmove_part = parts[4] if len(parts) > 4 else "0"
    fullmove_part = parts[5] if len(parts) > 5 else "1"

    expanded_board = re.sub(r"\d", lambda digit: "." * int(digit.group(0)), board_part)
    expanded_board = expanded_board.replace("/", "")

    if len(expanded_board) != 64:
        raise ValueError(f"Invalid partial board length: {len(expanded_board)} (expected 64)")

    tokens_list = []
    for char in expanded_board:
        if char == "?":
            tokens_list.append(mask_token)
        elif char == ".":
            tokens_list.append(FENTokens.no_piece)
        elif char in board_str_2_token:
            tokens_list.append(board_str_2_token[char])
        else:
            raise ValueError(f"Unknown board character: '{char}'")

    if side_part == "w":
        tokens_list.append(FENTokens.side_white)
    elif side_part == "b":
        tokens_list.append(FENTokens.side_black)
    elif "?" in side_part:
        tokens_list.append(mask_token)
    else:
        tokens_list.append(FENTokens.side_white)

    castling_chars = "KQkq"
    castling_tokens_map = [
        FENTokens.castle_white_king, FENTokens.castle_white_queen,
        FENTokens.castle_black_king, FENTokens.castle_black_queen
    ]
    if castling_part == "-":
        tokens_list.extend([FENTokens.no_castle] * 4)
    elif castling_part == "?":
        tokens_list.extend([mask_token] * 4)
    elif castling_part == "?-":
        tokens_list.extend([mask_token, mask_token, FENTokens.no_castle, FENTokens.no_castle])
    elif castling_part == "-?":
        tokens_list.extend([FENTokens.no_castle, FENTokens.no_castle, mask_token, mask_token])
    elif len(castling_part) == 4:
        for idx, (c, tok) in enumerate(zip(castling_chars, castling_tokens_map)):
            char = castling_part[idx]
            if char == c:
                tokens_list.append(tok)
            elif char == "?":
                tokens_list.append(mask_token)
            else:
                tokens_list.append(FENTokens.no_castle)
    else:
        has_q_mark = "?" in castling_part
        has_w_char = any(c in castling_part for c in "KQ")
        has_b_char = any(c in castling_part for c in "kq")
        
        if "K" in castling_part:
            tokens_list.append(FENTokens.castle_white_king)
        elif has_q_mark and not has_w_char:
            tokens_list.append(mask_token)
        else:
            tokens_list.append(FENTokens.no_castle)
            
        if "Q" in castling_part:
            tokens_list.append(FENTokens.castle_white_queen)
        elif has_q_mark and not has_w_char:
            tokens_list.append(mask_token)
        else:
            tokens_list.append(FENTokens.no_castle)

        if "k" in castling_part:
            tokens_list.append(FENTokens.castle_black_king)
        elif has_q_mark and not has_b_char:
            tokens_list.append(mask_token)
        else:
            tokens_list.append(FENTokens.no_castle)

        if "q" in castling_part:
            tokens_list.append(FENTokens.castle_black_queen)
        elif has_q_mark and not has_b_char:
            tokens_list.append(mask_token)
        else:
            tokens_list.append(FENTokens.no_castle)

    if "?" in ep_part:
        if len(ep_part) == 1:
            tokens_list.extend([mask_token, mask_token])
        else:
            f_tok = mask_token if ep_part[0] == "?" else enpassant_str_2_token.get(ep_part[0], FENTokens.none)
            r_tok = mask_token if ep_part[1] == "?" else enpassant_str_2_token.get(ep_part[1], FENTokens.none)
            tokens_list.extend([f_tok, r_tok])
    elif ep_part != "-" and len(ep_part) == 2:
        file_tok = enpassant_str_2_token.get(ep_part[0], FENTokens.none)
        rank_tok = enpassant_str_2_token.get(ep_part[1], FENTokens.none)
        tokens_list.extend([file_tok, rank_tok])
    else:
        tokens_list.extend([FENTokens.none, FENTokens.none])

    if "?" in halfmove_part:
        tokens_list.extend([mask_token, mask_token])
    else:
        hm_str = "." + halfmove_part if len(halfmove_part) == 1 else halfmove_part
        hm_toks = [counter_str_2_token.get(c, FENTokens.pad_counter) for c in hm_str[:2]]
        tokens_list.extend(hm_toks)

    if "?" in fullmove_part:
        tokens_list.extend([mask_token, mask_token, mask_token])
    else:
        fm_str = ".." + fullmove_part if len(fullmove_part) == 1 else ("." + fullmove_part if len(fullmove_part) == 2 else fullmove_part)
        fm_toks = [counter_str_2_token.get(c, FENTokens.pad_counter) for c in fm_str[:3]]
        tokens_list.extend(fm_toks)

    if config.predict_moves:
        if best_move is not None:
            move_toks = []
            for idx, c in enumerate(best_move):
                if c == "?":
                    move_toks.append(mask_token)
                elif idx < 4:
                    move_toks.append(enpassant_str_2_token.get(c, mask_token))
                else:
                    move_toks.append(promote_str_2_token.get(c, FENTokens.none))
            while len(move_toks) < 5:
                move_toks.append(FENTokens.none)
            tokens_list.extend(move_toks[:5])
        else:
            tokens_list.extend([mask_token] * 5)

    return torch.tensor(tokens_list, dtype=torch.long)

# ==========================================
# 3. Custom Diffusion Pipeline
# ==========================================

class ChessPuzzlePipeline(DiffusionPipeline):
    def __init__(self, model):
        super().__init__()
        self.register_modules(model=model)

    @torch.no_grad()
    def __call__(
        self,
        themes: str,
        rating: float,
        partial_board: str = None,
        best_move: str = None,
        batch_size: int = 1,
        steps: int = 256,
        temperature: float = 1.0,
        schedule: str = "linear",
        generate_move_last: bool = True,
        device: str = None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.model.to(device)
        self.model.eval()

        # 1. Preprocess context inputs (themes & rating)
        scaled_rating = scale_rating(rating)
        ratings_tensor = torch.full((batch_size, 1), scaled_rating, dtype=torch.float32, device=device)

        themes_tensor = torch.zeros((batch_size, len(unique_themes)), dtype=torch.float32, device=device)
        if themes:
            for t in themes.split():
                if t in theme_to_idx:
                    themes_tensor[:, theme_to_idx[t]] = 1.0

        # 2. Setup schedule
        if schedule == "linear":
            masking_schedule = LinearSchedule()
        elif schedule == "cosine":
            masking_schedule = CosineSchedule()
        elif schedule == "geometric":
            masking_schedule = GeometricSchedule()
        elif schedule == "polynomial":
            masking_schedule = PolynomialSchedule()
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        # 3. Handle optional partial board initialization
        mask_token = self.model.config.n_fen_tokens + (self.model.config.n_move_tokens if self.model.config.predict_moves else 0)
        seq_length = self.model.config.fen_length + (self.model.config.move_length if self.model.config.predict_moves else 0)

        if partial_board is not None:
            initial_tokens = partial_fen_to_tokens(partial_board, self.model.config, mask_token, best_move=best_move)
            tokens = initial_tokens.unsqueeze(0).repeat(batch_size, 1).to(device)
        else:
            tokens = torch.full((batch_size, seq_length), mask_token, device=device, dtype=torch.long)

        T_grid = torch.linspace(0, 1, steps + 1, device=device)

        if not self.model.config.predict_moves:
            generate_move_last = False
            
        if generate_move_last:
            phases = [
                (0, self.model.config.fen_length, steps), 
                (self.model.config.fen_length, seq_length, steps // 4)
            ]
        else:
            phases = [(0, seq_length, steps)]

        for start_idx, end_idx, step_count in phases:
            for i in range(step_count, 0, -1):
                t = T_grid[i]
                s = T_grid[i - 1]
                alpha_t = masking_schedule(t)
                alpha_s = masking_schedule(s)
                if s == 0.0:
                    alpha_s = torch.ones_like(alpha_s)
                
                logits = self.model(tokens, themes_tensor, ratings_tensor)
                probs = F.softmax(logits / temperature, dim=2)
                
                p_unmask = (alpha_s - alpha_t) / (1.0 - alpha_t + 1e-13)
                p_mask = (1.0 - alpha_s) / (1.0 - alpha_t + 1e-13)
                
                probs = torch.cat([
                    probs * p_unmask, 
                    torch.full((batch_size, seq_length, 1), p_mask, device=device, dtype=probs.dtype)
                ], dim=2)

                # Gumbel-max sampling
                log_probs = torch.log(probs + 1e-13)
                u = torch.rand_like(log_probs)
                gumbel_noise = -torch.log(-torch.log(u + 1e-13) + 1e-13)
                new_samples = torch.argmax(log_probs + gumbel_noise, dim=-1)
                
                is_masked = (tokens == mask_token)
                in_window = torch.zeros_like(is_masked, dtype=torch.bool)
                in_window[:, start_idx:end_idx] = True
                is_updatable = is_masked & in_window

                tokens = torch.where(is_updatable, new_samples, tokens)

        # 4. Detokenize results
        results = []
        tokens_cpu = tokens.cpu().tolist()
        for idx in range(batch_size):
            fen = tokens_to_fen(tokens_cpu[idx][:self.model.config.fen_length])
            move = tokens_to_move(tokens_cpu[idx][self.model.config.fen_length:]) if self.model.config.predict_moves else None
            results.append({"fen": fen, "move": move})
            
        return results
