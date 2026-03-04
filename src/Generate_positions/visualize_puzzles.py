import pandas as pd
import chess
import chess.svg
import chess.engine
import ast
import base64


CSV_FILE = "./generated_puzzles_lichess_dataset_theme_distribution_correct_ratings.csv"
OUTPUT_HTML = "puzzles_output.html"
STOCKFISH_PATH = "./../../Stockfish/src/stockfish"
ANALYSIS_TIME_LIMIT = 0.5

def get_mainline(board, engine_path, time_limit):
    """Compute the main line (PV) using Stockfish."""
    try:
        with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
            info = engine.analyse(board, chess.engine.Limit(time=time_limit))
            
            # Format the Principal Variation (PV) into standard algebraic notation (SAN)
            if "pv" in info:
                pv_moves = info["pv"]
                # Create a copy of the board to generate SAN strings
                temp_board = board.copy()
                san_moves = []
                for move in pv_moves[:5]: # limit to top 5 moves for the main line
                    san_moves.append(temp_board.san(move))
                    temp_board.push(move)
                return " ".join(san_moves)
            return "No line found"
    except Exception as e:
        return f"Engine error (ensure Stockfish is installed): {e}"

def generate_html():
    # 1. Read the CSV
    df = pd.read_csv(CSV_FILE)
    
    # 2. Filter the data: Legal, Unique Solution (is_puzzle), Counter-intuitive
    # Note: Using boolean checking. Ensure the columns are booleans or strings based on your CSV parsing
    filtered_df = df[
        (df['is_legal'] == True) & 
        (df['is_puzzle'] == True) & 
        (df['counter_intuitive'] == True)
    ].copy()
    
    if filtered_df.empty:
        print("No puzzles matched the filter criteria!")
        # For testing purposes, you might want to comment out the filter to see output.
    
    html_cards = []
    
    # 3. Process each puzzle
    for index, row in filtered_df.iterrows():
        fen = row['fen']
        rating = row['target_rating']
        
        # Safely evaluate the string representation of the list
        try:
            themes = ast.literal_eval(row['target_themes'])
        except:
            themes = []
            
        # Parse board
        board = chess.Board(fen)
        
        # Compute mainline
        mainline = get_mainline(board, STOCKFISH_PATH, ANALYSIS_TIME_LIMIT)
        
        # Generate SVG board representation
        svg_data = chess.svg.board(board=board, size=300)
        # Base64 encode the SVG to embed directly in the HTML img tag
        b64_svg = base64.b64encode(svg_data.encode('utf-8')).decode('utf-8')
        
        # Format theme pills
        theme_html = "".join([f'<span class="theme-pill">{theme}</span>' for theme in themes])
        
        # Who to move?
        to_move = "White to move" if board.turn == chess.WHITE else "Black to move"
        
        # Build the HTML card
        card = f"""
        <div class="card">
            <div class="board">
                <img src="data:image/svg+xml;base64,{b64_svg}" alt="Chess Board" />
            </div>
            <div class="info">
                <h3>{to_move}</h3>
                <p><strong>Rating:</strong> {rating}</p>
                <p><strong>FEN:</strong> <span class="fen-text">{fen}</span></p>
                <p><strong>Main Line:</strong> <span class="main-line">{mainline}</span></p>
                <div class="themes">
                    {theme_html}
                </div>
            </div>
        </div>
        """
        html_cards.append(card)

    # 4. Construct the final HTML page with embedded CSS
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Filtered Chess Puzzles</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f4f7f6;
                color: #333;
                margin: 0;
                padding: 20px;
            }}
            h1 {{
                text-align: center;
                color: #2c3e50;
            }}
            .container {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
                gap: 20px;
                max-width: 1200px;
                margin: 0 auto;
            }}
            .card {{
                background-color: #fff;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                overflow: hidden;
                display: flex;
                flex-direction: column;
                transition: transform 0.2s;
            }}
            .card:hover {{
                transform: translateY(-5px);
            }}
            .board {{
                background-color: #eaeaea;
                display: flex;
                justify-content: center;
                padding: 15px;
            }}
            .info {{
                padding: 20px;
            }}
            .info h3 {{
                margin-top: 0;
                color: #2980b9;
            }}
            .fen-text, .main-line {{
                background: #ecf0f1;
                padding: 2px 6px;
                border-radius: 4px;
                font-family: monospace;
                font-size: 0.9em;
                word-break: break-all;
            }}
            .main-line {{
                background: #e8f5e9;
                color: #2e7d32;
                font-weight: bold;
            }}
            .themes {{
                margin-top: 15px;
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
            }}
            .theme-pill {{
                background-color: #34495e;
                color: #fff;
                padding: 5px 10px;
                border-radius: 15px;
                font-size: 0.8em;
                text-transform: capitalize;
            }}
        </style>
    </head>
    <body>
        <h1>Counter-Intuitive Puzzles</h1>
        <div class="container">
            {"".join(html_cards)}
        </div>
    </body>
    </html>
    """

    # 5. Save to file
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_template)
    
    print(f"Successfully generated {OUTPUT_HTML} with {len(html_cards)} puzzles.")

if __name__ == "__main__":
    generate_html()
