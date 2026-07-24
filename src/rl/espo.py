import torch
import pandas as pd

import random


state_of_game_tokens = ("opening", "middlegame", "endgame")
endgames = ("pawnEndgame", "bishopEndgame", "knightEndgame", "rookEndgame", "queenEndgame", "queenRookEndgame")

is_mate = "mate"
mate_lengths = ("mateIn1", "mateIn2", "mateIn3", "mateIn4", "mateIn5")
types_of_mate = ("backRankMate", "bodenMate", "smotheredMate", "hookMate", "doubleBishopMate", "arabianMate", "dovetailMate", "anastasiaMate")

lengths = ("oneMove", "short", "long", "veryLong")

winnings = ("crushing", "advantage")
other = ("hangingPiece", "fork", "interference", "kingsideAttack", "zugzwang", "exposedKing", "skewer", "pin", "quietMove", "discoveredAttack", "sacrifice", "deflection", "advancedPawn", "attraction", "promotion", "queensideAttack", "defensiveMove", "attackingF2F7", "clearance", "intermezzo", "equality", "trappedPiece", "xRayAttack", "capturingDefender", "doubleCheck", "enPassant", "castling", "underPromotion")

dataset = None
def generate_random_themes(batch_size, lichess_distribution=False):
    global dataset
    if lichess_distribution:
        if dataset is None:
            dataset = pd.read_csv("./src/dataset/dataset.csv")
        rows = dataset.sample(n=batch_size)
        themes = rows["Themes"].str.split(" ").to_list()
        ratings = torch.from_numpy(rows["Rating"].to_numpy())
    else:
        themes = []
        for _ in range(batch_size):
            position_themes = [random.choice(lengths)]
            state_of_game = random.choice(state_of_game_tokens)
            position_themes.append(state_of_game)

            if state_of_game == "endgame":
                position_themes.append(random.choice(endgames))
            
            if torch.rand(1) < 0.1:
                position_themes.append(is_mate)
                position_themes.append(random.choice(mate_lengths))
                position_themes.append(random.choice(types_of_mate))
            else:
                position_themes.append(random.choice(winnings))
                position_themes.append(random.choice(other))

            themes.append(position_themes)
        
        ratings = 3000 * torch.rand((batch_size,)) + 300

    return themes, ratings

def theme_reward(base_themes, puzzle_themes):
    base_set = set(base_themes)
    puzzle_set = set(puzzle_themes)

    base_state = base_set.intersection(state_of_game_tokens)
    if not base_state.issubset(puzzle_set):
        return False

    if "endgame" in base_set:
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
        base_other = base_set.intersection(other)
        if not base_other.issubset(puzzle_set):
            return False

    return True
