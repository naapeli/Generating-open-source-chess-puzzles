from stockfish import Stockfish

import math


stockfish = Stockfish(path="./Stockfish/src/stockfish", depth=15)


def calculate_winning_chances(cp):
    # from https://lichess.org/page/accuracy and https://arxiv.org/pdf/2402.04494
    MULTIPLIER = 0.00368208
    return 2 / (1 + math.exp(-MULTIPLIER * cp)) - 1

def legal(stockfish, fen):
    return stockfish.is_fen_valid(fen)  # according to the docs, returns false even if the position is legal if no move can be made (checkmate etc)

def unique(stockfish, fen):
    stockfish.set_fen_position(fen)
    moves = stockfish.get_top_moves(1, verbose=True)
    print(moves[0].keys())
    print(moves[0].values())
    # print(moves[0]["PVMoves"])
    return _unique(stockfish, fen)

def _unique(stockfish, fen):  # assumes stockfish returns evaluations as positive for the current side to move in the fen
    TAU_UNI = 0.5

    stockfish.set_fen_position(fen)
    moves = stockfish.get_top_moves(2, verbose=True)

    if moves[0]["Mate"] is not None:
        if moves[0]["Mate"] > 0:
            pass
    else:
        if moves[1]["Mate"] is not None: raise NotImplementedError()
        if calculate_winning_chances(moves[0]["Centipawn"]) - calculate_winning_chances(moves[1]["Centipawn"]) < TAU_UNI: return False  # if this position is not unique, return False, otherwise continue with the next move in the principal variation



    
    return False

def counter_intuitiveness(stockfish, fen):
    stockfish.set_fen_position(fen)
    stockfish.get_top_moves(2, verbose=True)

    raise NotImplementedError()


print(unique(stockfish, "2r3k1/2r3Pp/8/3p2P1/1pn1p1N1/p3q3/PPP2Q2/1K3R1R b - - 4 34"))
print(unique(stockfish, "4k3/8/4q2p/8/1p3p2/2b2P1B/7P/5R1K w - - 0 51"))
print(unique(stockfish, "8/6p1/7p/4p2P/5PP1/3k4/5K2/8 b - - 0 52"))
