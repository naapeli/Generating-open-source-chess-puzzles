state_of_game_tokens = ("opening", "middlegame", "endgame")
endgames = ("pawnEndgame", "bishopEndgame", "knightEndgame", "rookEndgame", "queenEndgame", "queenRookEndgame")

is_mate = "mate"
mate_lengths = ("mateIn1", "mateIn2", "mateIn3", "mateIn4", "mateIn5")
types_of_mate = ("backRankMate", "bodenMate", "smotheredMate", "hookMate", "doubleBishopMate", "arabianMate", "dovetailMate", "anastasiaMate")

lengths = ("oneMove", "short", "long", "veryLong")

winnings = ("crushing", "advantage")
other = ("hangingPiece", "fork", "interference", "kingsideAttack", "zugzwang", "exposedKing", "skewer", "pin", "quietMove", "discoveredAttack", "sacrifice", "deflection", "advancedPawn", "attraction", "promotion", "queensideAttack", "defensiveMove", "attackingF2F7", "clearance", "intermezzo", "equality", "trappedPiece", "xRayAttack", "capturingDefender", "doubleCheck", "enPassant", "castling", "underPromotion")

def theme_reward(base_themes, puzzle_themes):
    base_set = set(base_themes)
    puzzle_set = set(puzzle_themes)

    base_state = base_set.intersection(state_of_game_tokens)
    if not base_state.issubset(puzzle_set):
        return False

    base_endgame = base_set.intersection(endgames)
    if not base_endgame.issubset(puzzle_set):
        return False
    
    if "mate" in base_set:
        if "mate" not in puzzle_set:
            return False
            
        base_mate_length = base_set.intersection(mate_lengths)
        if not base_mate_length.issubset(puzzle_set):
            return False
            
        base_mate_type = base_set.intersection(types_of_mate)
        if not base_mate_type.issubset(puzzle_set):
            return False
    else:
        if "mate" in puzzle_set:  # if model generated a checkmate when we did not ask it to do so, return False
            return False
            
    base_other = base_set.intersection(other)
    if not base_other.issubset(puzzle_set):
        return False

    return True
