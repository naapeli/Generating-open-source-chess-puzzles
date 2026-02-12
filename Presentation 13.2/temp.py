import chess
import chess.svg

# The FEN you provided
fen = "8/5pp1/8/2K2Pp1/7P/6k1/8/8 w - - 0 48"
board = chess.Board(fen)


# Generate SVG with the critical move highlighted
boardsvg = chess.svg.board(
    board=board,
    size=400
)

# Save to file
with open("Google presentation/figures/endgame_setup.svg", "w") as f:
    f.write(boardsvg)
