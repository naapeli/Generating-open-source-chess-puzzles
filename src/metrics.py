from stockfish import Stockfish

import math


def calculate_winning_chances(cp):
    # from https://lichess.org/page/accuracy and https://arxiv.org/pdf/2402.04494 and other papers
    MULTIPLIER = 0.00368208
    return 2 / (1 + math.exp(-MULTIPLIER * cp)) - 1

def legal(stockfish: Stockfish, fen):
    return stockfish.is_fen_valid(fen)  # according to the docs, returns false even if the position is legal if no move can be made (checkmate etc)

def unique(stockfish: Stockfish, fen):  # assumes stockfish returns evaluations as positive for the current side to move in the fen
    stockfish.set_fen_position(fen)
    return _unique(stockfish)

def _unique(stockfish: Stockfish):
    TAU_UNI = 0.5

    moves = stockfish.get_top_moves(2)

    if len(moves) < 2:
        return True

    best_move = moves[0]
    second_move = moves[1]

    is_unique_step = False
    if best_move["Mate"] is not None:
        if best_move["Mate"] > 0:
            if second_move["Mate"] is not None and second_move["Mate"] > 0:
                return False 
            is_unique_step = True
        else:
            return False

    else:
        if (calculate_winning_chances(best_move["Centipawn"]) - calculate_winning_chances(second_move["Centipawn"])) >= TAU_UNI:
            is_unique_step = True
        else:
            return False

    if is_unique_step:
        stockfish.make_moves_from_current_position([best_move["Move"]])
        
        # Check for immediate game end (e.g., Checkmate)
        # Note: Stockfish wrapper might not update "Mate" status instantly without info.
        # But if we mated, get_best_move() usually returns None or we can check legal moves.
        if best_move["Mate"] == 1: 
            return True # Mate delivered, puzzle over.

        opponent_response = stockfish.get_best_move()        
        if opponent_response is None:
            return True

        stockfish.make_moves_from_current_position([opponent_response])        
        return _unique(stockfish)

    return False


def counter_intuitiveness(stockfish: Stockfish, fen):
    stockfish.set_fen_position(fen)
    stockfish.get_top_moves(2, verbose=True)

    raise NotImplementedError()


# print(unique(stockfish, "2r3k1/2r3Pp/8/3p2P1/1pn1p1N1/p3q3/PPP2Q2/1K3R1R b - - 4 34"))
# print(unique(stockfish, "4k3/8/4q2p/8/1p3p2/2b2P1B/7P/5R1K w - - 0 51"))
# print(unique(stockfish, "8/6p1/7p/4p2P/5PP1/3k4/5K2/8 b - - 0 52"))
