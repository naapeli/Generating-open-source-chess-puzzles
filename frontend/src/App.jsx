import { useState, useEffect, useRef } from 'react';
import { Chess } from 'chess.js';
import { Chessboard } from 'react-chessboard';
import { generatePuzzle, checkPuzzleStatus, cancelPuzzleGeneration } from './api/puzzleApi';
import './App.css';

const CATEGORIES = {
  "Game Phase": ["opening", "middlegame", "endgame"],
  "Endgames": ["pawnEndgame", "bishopEndgame", "knightEndgame", "rookEndgame", "queenEndgame", "queenRookEndgame"],
  "Mates": ["mate", "mateIn1", "mateIn2", "mateIn3", "mateIn4", "mateIn5", "backRankMate", "bodenMate", "smotheredMate", "hookMate", "doubleBishopMate", "arabianMate", "dovetailMate", "anastasiaMate"],
  "Lengths": ["oneMove", "short", "long", "veryLong"],
  "Outcomes": ["crushing", "advantage"],
  "Tactical Themes": ["hangingPiece", "fork", "interference", "kingsideAttack", "zugzwang", "exposedKing", "skewer", "pin", "quietMove", "discoveredAttack", "sacrifice", "deflection", "advancedPawn", "attraction", "promotion", "queensideAttack", "defensiveMove", "attackingF2F7", "clearance", "intermezzo", "equality", "trappedPiece", "xRayAttack", "capturingDefender", "doubleCheck", "enPassant", "castling", "underPromotion"]
};

// Helper to format theme names for display
function formatThemeName(theme) {
  if (!theme) return '';

  const customNames = {
    "xRayAttack": "X-Ray Attack",
    "attackingF2F7": "Attacking f2/f7",
    "underPromotion": "Underpromotion",
  };

  if (customNames[theme]) {
    return customNames[theme];
  }

  // Generic formatter: e.g., pawnEndgame -> Pawn Endgame, mateIn1 -> Mate In 1
  let formatted = theme.charAt(0).toUpperCase() + theme.slice(1);
  formatted = formatted.replace(/([a-z0-9])([A-Z])/g, '$1 $2');
  formatted = formatted.replace(/([a-zA-Z])([0-9]+)/g, '$1 $2');
  formatted = formatted.replace(/([0-9]+)([a-zA-Z])/g, '$1 $2');

  return formatted;
}

// Helper to parse moves in abstract/UCI notation (e.g. "e2e4" -> { from: "e2", to: "e4", promotion: undefined })
function parseMove(moveStr) {
  if (typeof moveStr === 'string' && moveStr.length >= 4) {
    const from = moveStr.slice(0, 2);
    const to = moveStr.slice(2, 4);
    const promotion = moveStr.length > 4 ? moveStr.charAt(4) : undefined;
    return { from, to, promotion };
  }
  return moveStr; // fallback to SAN
}

// Helper to pre-process mainline moves and split any Python-concatenated strings (e.g. "Kh1Rxh2+")
function preprocessMainline(moves) {
  if (!moves) return [];
  const result = [];
  const regex = /(?:[KQRBN]?[a-h1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|O-O-O|O-O)/g;
  for (const move of moves) {
    if (typeof move === 'string') {
      const matches = move.match(regex);
      if (matches) {
        result.push(...matches);
      } else {
        result.push(move);
      }
    } else {
      result.push(move);
    }
  }
  return result;
}

function App() {
  const [game, setGame] = useState(new Chess());
  const [gameFen, setGameFen] = useState(game.fen());
  const [rating, setRating] = useState(1500);

  const [openCategory, setOpenCategory] = useState(null);

  // Asynchronous Job Status
  const [activeJobId, setActiveJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState('idle'); // 'idle' | 'generating' | 'completed' | 'failed' | 'cancelled'
  const [error, setError] = useState(null);
  const [puzzleDetails, setPuzzleDetails] = useState(null);
  const [boardOrientation, setBoardOrientation] = useState("white");

  // Puzzle Solving State Machine
  const [currentMoveIndex, setCurrentMoveIndex] = useState(0);
  const [puzzleSolved, setPuzzleSolved] = useState(false);
  const [puzzleFeedback, setPuzzleFeedback] = useState(null); // 'correct' | 'incorrect' | 'completed' | null
  const [hasFailed, setHasFailed] = useState(false);
  const [showSolutionForce, setShowSolutionForce] = useState(false);

  // Keep a mutable ref of activeJobId so beforeunload event listener can see its latest value
  const jobIdRef = useRef(activeJobId);
  useEffect(() => {
    jobIdRef.current = activeJobId;
  }, [activeJobId]);

  // Tell backend to cancel if tab/window is closed during generation
  useEffect(() => {
    function handleBeforeUnload() {
      if (jobIdRef.current) {
        cancelPuzzleGeneration(jobIdRef.current);
      }
    }
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, []);

  // Poll status of active generation job
  useEffect(() => {
    if (!activeJobId || jobStatus !== 'generating') return;

    let intervalId;

    async function poll() {
      try {
        const res = await checkPuzzleStatus(activeJobId);
        if (res.status === 'completed' && res.puzzle) {
          setJobStatus('completed');
          setActiveJobId(null);

          // Preprocess mainline to handle any concatenated moves (e.g. "Kh1Rxh2+")
          const processedPuzzle = {
            ...res.puzzle,
            mainline: preprocessMainline(res.puzzle.mainline)
          };
          setPuzzleDetails(processedPuzzle);

          // Reset puzzle solving state machine
          setPuzzleSolved(false);
          setPuzzleFeedback(null);
          setCurrentMoveIndex(0);
          setHasFailed(false);
          setShowSolutionForce(false);

          // Set board to puzzle position
          const newGame = new Chess(processedPuzzle.position.fen);
          setGame(newGame);
          setGameFen(newGame.fen());

          // Auto-flip board to the perspective of the solver (active turn of the loaded position)
          const solverColor = newGame.turn() === 'b' ? 'black' : 'white';
          setBoardOrientation(solverColor);

          clearInterval(intervalId);
        } else if (res.status === 'failed') {
          setJobStatus('failed');
          setActiveJobId(null);
          setError('Puzzle generation failed.');
          clearInterval(intervalId);
        } else if (res.status === 'cancelled') {
          setJobStatus('cancelled');
          setActiveJobId(null);
          clearInterval(intervalId);
        }
      } catch (err) {
        console.error("Polling error:", err);
      }
    }

    intervalId = setInterval(poll, 1000);
    poll(); // initial check

    return () => {
      clearInterval(intervalId);
    };
  }, [activeJobId, jobStatus]);

  const [selectedThemes, setSelectedThemes] = useState(() => {
    const initial = {};
    Object.values(CATEGORIES).flat().forEach(theme => {
      initial[theme] = false;
    });
    return initial;
  });

  // Function to handle make move logic
  function makeAMove(move) {
    try {
      const gameCopy = new Chess(game.fen());
      const result = gameCopy.move(move);

      // If move is valid, update game object and state
      if (result) {
        setGame(gameCopy);
        setGameFen(gameCopy.fen());
        return result;
      }
    } catch (e) {
      // Catch exceptions from chess.js on illegal moves
      return null;
    }
    return null;
  }

  // Handle drag-and-drop moves
  function onDrop(sourceSquare, targetSquare) {
    // If a puzzle is active, we check if the user's move matches the expected move
    if (puzzleDetails) {
      if (puzzleSolved) return false;
      if (currentMoveIndex % 2 !== 0) return false; // opponent's turn to play (odd indices belong to opponent)

      const expectedMove = puzzleDetails.mainline[currentMoveIndex];

      // Determine if the user's move matches the expected SAN move by trying to execute it on a temp board
      const tempGame = new Chess(game.fen());
      let expectedResult = null;
      try {
        expectedResult = tempGame.move(expectedMove);
      } catch (e) {
        try {
          const cleanExpected = expectedMove.replace(/[+#]/g, '');
          expectedResult = tempGame.move(cleanExpected);
        } catch (e2) {
          console.error("Could not parse expected SAN move:", expectedMove, e2);
        }
      }

      if (!expectedResult) {
        // Fallback to checking basic coordinates if SAN parse fails
        const parsedExpected = parseMove(expectedMove);
        const isCorrectSquare = sourceSquare === parsedExpected.from && targetSquare === parsedExpected.to;
        if (!isCorrectSquare) {
          setPuzzleFeedback('incorrect');
          setHasFailed(true);
          setTimeout(() => {
            setPuzzleFeedback(prev => prev === 'incorrect' ? null : prev);
          }, 1500);
          return false;
        }
      } else {
        const isCorrect = sourceSquare === expectedResult.from && targetSquare === expectedResult.to;
        if (!isCorrect) {
          setPuzzleFeedback('incorrect');
          setHasFailed(true);
          setTimeout(() => {
            setPuzzleFeedback(prev => prev === 'incorrect' ? null : prev);
          }, 1500);
          return false;
        }
      }

      // Move is correct! Apply it on the board
      const gameCopy = new Chess(game.fen());
      try {
        const result = gameCopy.move({
          from: sourceSquare,
          to: targetSquare,
          promotion: (expectedResult && expectedResult.promotion) || 'q'
        });
        if (!result) return false;

        setGame(gameCopy);
        setGameFen(gameCopy.fen());

        const nextIndex = currentMoveIndex + 1;

        if (nextIndex === puzzleDetails.mainline.length) {
          setPuzzleSolved(true);
          setShowSolutionForce(true);
          setPuzzleFeedback('completed');
          setCurrentMoveIndex(nextIndex);
        } else {
          setPuzzleFeedback('correct');
          setCurrentMoveIndex(nextIndex);

          // Play opponent response automatically after 800ms
          setTimeout(() => {
            const opponentReply = puzzleDetails.mainline[nextIndex];
            const opponentGame = new Chess(gameCopy.fen());
            try {
              opponentGame.move(opponentReply);
            } catch (e) {
              try {
                const cleanReply = opponentReply.replace(/[+#]/g, '');
                opponentGame.move(cleanReply);
              } catch (e2) {
                console.error("Could not execute opponent reply SAN move:", opponentReply, e2);
              }
            }

            setGame(opponentGame);
            setGameFen(opponentGame.fen());

            const postOpponentIndex = nextIndex + 1;
            setCurrentMoveIndex(postOpponentIndex);
            setPuzzleFeedback(null);

            // Check if opponent's move was the last move (e.g. odd mainline lengths)
            if (postOpponentIndex === puzzleDetails.mainline.length) {
              setPuzzleSolved(true);
              setShowSolutionForce(true);
              setPuzzleFeedback('completed');
            }
          }, 800);
        }
        return true;
      } catch (err) {
        console.error("Error executing user move:", err);
        return false;
      }
    } else {
      const move = makeAMove({
        from: sourceSquare,
        to: targetSquare,
        promotion: 'q',
      });
      if (move === null) return false;
      return true;
    }
  }

  // Reset the game to the puzzle position (or starting position if no puzzle is active)
  function handleReset() {
    if (puzzleDetails) {
      setPuzzleSolved(false);
      setPuzzleFeedback(null);
      setCurrentMoveIndex(0);

      const newGame = new Chess(puzzleDetails.position.fen);
      setGame(newGame);
      setGameFen(newGame.fen());

      const solverColor = newGame.turn() === 'b' ? 'black' : 'white';
      setBoardOrientation(solverColor);
    } else {
      const newGame = new Chess();
      setGame(newGame);
      setGameFen(newGame.fen());
      setBoardOrientation("white");
    }
  }

  // Flip the board orientation
  function handleFlipBoard() {
    setBoardOrientation(prev => prev === 'white' ? 'black' : 'white');
  }

  // Toggle category visibility (only one can be open at a time)
  function toggleCategory(catName) {
    setOpenCategory(prev => prev === catName ? null : catName);
  }

  // Toggle a tactical theme selection
  function toggleTheme(themeName) {
    setSelectedThemes(prev => {
      const mateRelatedLengths = ["mateIn1", "mateIn2", "mateIn3", "mateIn4", "mateIn5"];
      const activeMateLength = mateRelatedLengths.find(t => prev[t]);

      const MATE_TO_PUZZLE_LENGTH = {
        mateIn1: "oneMove",
        mateIn2: "short",
        mateIn3: "long",
        mateIn4: "veryLong",
        mateIn5: "veryLong"
      };

      // If a checkmate length is active, check if the user is trying to change the puzzle length
      const puzzleLengths = ["oneMove", "short", "long", "veryLong"];
      if (activeMateLength && puzzleLengths.includes(themeName)) {
        // Prevent manual unticking or changing puzzle length while checkmate length is active
        return prev;
      }

      // 1. Basic toggle
      const next = { ...prev, [themeName]: !prev[themeName] };

      // 2. Enforce mutual exclusivity (max 1 selected) for specific groups
      const EXCLUSIVE_GROUPS = [
        ["opening", "middlegame", "endgame"],
        ["pawnEndgame", "bishopEndgame", "knightEndgame", "rookEndgame", "queenEndgame", "queenRookEndgame"],
        ["mateIn1", "mateIn2", "mateIn3", "mateIn4", "mateIn5"],
        ["backRankMate", "bodenMate", "smotheredMate", "hookMate", "doubleBishopMate", "arabianMate", "dovetailMate", "anastasiaMate"],
        ["oneMove", "short", "long", "veryLong"],
        ["crushing", "advantage"]
      ];

      if (next[themeName]) {
        const group = EXCLUSIVE_GROUPS.find(g => g.includes(themeName));
        if (group) {
          group.forEach(t => {
            if (t !== themeName) {
              next[t] = false;
            }
          });
        }
      }

      // 3. Correlate checkmate length and puzzle length
      if (mateRelatedLengths.includes(themeName) && next[themeName]) {
        const correspondingPuzzleLength = MATE_TO_PUZZLE_LENGTH[themeName];
        next[correspondingPuzzleLength] = true;
        // Make sure all other puzzle lengths are unchecked
        puzzleLengths.forEach(pl => {
          if (pl !== correspondingPuzzleLength) {
            next[pl] = false;
          }
        });
      }

      // 4. Force parent "mate" base theme if any child is active
      const mateRelatedTypes = [
        "mateIn1", "mateIn2", "mateIn3", "mateIn4", "mateIn5",
        "backRankMate", "bodenMate", "smotheredMate", "hookMate",
        "doubleBishopMate", "arabianMate", "dovetailMate", "anastasiaMate"
      ];
      const anyMateRelatedActive = mateRelatedTypes.some(t => next[t]);

      if (anyMateRelatedActive) {
        next["mate"] = true;
      }
      if (themeName === "mate" && !next["mate"] && anyMateRelatedActive) {
        next["mate"] = true;
      }

      // 5. Force parent "endgame" base theme if any child is active
      const endgameRelated = [
        "pawnEndgame", "bishopEndgame", "knightEndgame",
        "rookEndgame", "queenEndgame", "queenRookEndgame"
      ];
      const anyEndgameRelatedActive = endgameRelated.some(t => next[t]);

      if (anyEndgameRelatedActive) {
        next["endgame"] = true;
      }
      if (themeName === "endgame" && !next["endgame"] && anyEndgameRelatedActive) {
        next["endgame"] = true;
      }

      return next;
    });
  }

  // Clear all selected themes at once
  function clearAllThemes() {
    setSelectedThemes(prev => {
      const next = {};
      Object.keys(prev).forEach(theme => {
        next[theme] = false;
      });
      return next;
    });
  }


  // Handle start generation job request
  async function handleGenerate() {
    if (jobStatus === 'generating') return;

    const activeThemes = Object.keys(selectedThemes).filter(theme => selectedThemes[theme]);

    setJobStatus('generating');
    setError(null);
    setPuzzleDetails(null);
    setPuzzleSolved(false);
    setPuzzleFeedback(null);
    setCurrentMoveIndex(0);
    setHasFailed(false);
    setShowSolutionForce(false);

    try {
      const res = await generatePuzzle({
        rating,
        themes: activeThemes,
      });
      setActiveJobId(res.jobId);
    } catch (err) {
      console.error("Generate request failed:", err);
      setJobStatus('failed');
      setError('Failed to initiate puzzle generation. Is the backend server running?');
    }
  }

  // Handle manual cancel request
  async function handleCancel() {
    if (!activeJobId) return;
    const currentId = activeJobId;
    setActiveJobId(null);
    setJobStatus('cancelled');
    try {
      await cancelPuzzleGeneration(currentId);
    } catch (err) {
      console.error("Error sending cancel request:", err);
    }
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1 className="app-title">Chess Puzzle Generator</h1>
        <p className="app-subtitle">
          Generate and solve artificially generated chess puzzles.
        </p>
      </header>

      <main className="main-content">
        {/* Chessboard Column */}
        <section className="chessboard-section">
          <div className="chessboard-wrapper">
            <Chessboard
              position={gameFen}
              onPieceDrop={onDrop}
              boardOrientation={boardOrientation}
              customBoardStyle={{
                borderRadius: '8px',
                boxShadow: '0 5px 15px rgba(0, 0, 0, 0.5)',
              }}
            />
          </div>
          <div className="board-actions">
            {jobStatus === 'generating' ? (
              <div className="loading-overlay">
                <div className="spinner"></div>
                <div className="loading-text">Generating Puzzle</div>
                <div className="loading-subtext">Running AI model and filtering positions</div>
                <button className="btn-cancel" onClick={handleCancel}>Cancel Job</button>
              </div>
            ) : (
              <>
                <button
                  id="reset-board-btn"
                  className="btn-secondary"
                  onClick={handleReset}
                  disabled={jobStatus === 'generating'}
                >
                  Reset Board
                </button>
                <button
                  id="flip-board-btn"
                  className="btn-secondary"
                  onClick={handleFlipBoard}
                  disabled={jobStatus === 'generating'}
                >
                  Flip Board
                </button>
              </>
            )}
          </div>

          {/* Active Turn and FEN Indicator */}
          {puzzleDetails && (() => {
            const startTurn = puzzleDetails.position.fen.split(' ')[1];
            return (
              <div className="board-status-info">
                <div className="turn-indicator">
                  <span className={`turn-dot ${startTurn === 'w' ? 'white-turn' : 'black-turn'}`}></span>
                  <span className="turn-text">
                    {startTurn === 'w' ? 'White to move' : 'Black to move'}
                  </span>
                </div>

                {puzzleFeedback && (
                  <div className={`puzzle-feedback-banner ${puzzleFeedback}`}>
                    {puzzleFeedback === 'correct' && 'Correct! Keep going...'}
                    {puzzleFeedback === 'incorrect' && 'Incorrect move. Try again!'}
                    {puzzleFeedback === 'completed' && 'Completed! Puzzle Solved.'}
                  </div>
                )}

                <div className="fen-display-container">
                  <span className="fen-label">FEN:</span>
                  <input
                    type="text"
                    readOnly
                    value={gameFen}
                    className="fen-input"
                    onClick={(e) => e.target.select()}
                    title="Click to select FEN"
                  />
                </div>
              </div>
            );
          })()}
        </section>

        {/* Dynamic Control Panel Column */}
        <section className="control-panel-section">
          <div className="control-panel-card">
            <header className="panel-header">
              <div className="panel-header-row">
                <h2 className="panel-title">Puzzle Controls</h2>
                {Object.values(selectedThemes).some(Boolean) && (
                  <button
                    className="btn-clear-all"
                    onClick={clearAllThemes}
                    title="Clear all selected themes"
                  >
                    Clear All
                  </button>
                )}
              </div>
            </header>

            <div className="panel-body">
              {/* Target Rating Setting and Generate Trigger */}
              <div className="generator-settings">
                <div className="setting-row">
                  <label className="setting-label" htmlFor="rating-slider-input">
                    <span>Target Difficulty</span>
                    <span className="setting-value">{rating} ELO</span>
                  </label>
                  <input
                    id="rating-slider-input"
                    type="range"
                    min="800"
                    max="2800"
                    step="50"
                    value={rating}
                    onChange={(e) => setRating(Number(e.target.value))}
                    className="rating-slider"
                    disabled={jobStatus === 'generating'}
                  />
                </div>


                <button
                  id="generate-puzzles-btn"
                  className="btn-generate"
                  onClick={handleGenerate}
                  disabled={jobStatus === 'generating'}
                >
                  {jobStatus === 'generating' ? 'Generating...' : 'Generate Puzzle'}
                </button>

                {error && (
                  <div style={{ color: '#f87171', fontSize: '0.85rem', marginTop: '0.5rem', textAlign: 'center' }}>
                    {error}
                  </div>
                )}
              </div>

              {/* Generated Puzzle Info Card */}
              {puzzleDetails && (puzzleSolved || showSolutionForce) ? (
                <div className="puzzle-details-card">
                  <h3 className="puzzle-details-title">Puzzle Solution</h3>
                  {puzzleDetails.mainline && puzzleDetails.mainline.length > 0 && (
                    <div className="mainline-container" style={{ marginTop: 0 }}>
                      <span className="detail-label">Solution Sequence</span>
                      <div className="mainline-moves" style={{ fontSize: '1.05rem' }}>
                        {puzzleDetails.mainline.join(' → ')}
                      </div>
                    </div>
                  )}
                </div>
              ) : puzzleDetails ? (
                <div className="puzzle-details-card info-placeholder">
                  <span className="placeholder-icon">🎯</span>
                  <span className="placeholder-text">Solve the puzzle to reveal the solution!</span>
                  {hasFailed && (
                    <button
                      className="btn-secondary"
                      onClick={() => setShowSolutionForce(true)}
                      style={{ marginTop: '0.75rem', width: 'auto', padding: '0.5rem 1rem' }}
                    >
                      Reveal Solution
                    </button>
                  )}
                </div>
              ) : null}

              {/* Collapsible Accordion Sections for Categories */}
              {Object.keys(CATEGORIES).map(catName => {
                const themes = CATEGORIES[catName];
                const activeCount = themes.filter(t => selectedThemes[t]).length;
                const isExpanded = openCategory === catName;

                return (
                  <div
                    key={catName}
                    className={`theme-accordion ${isExpanded ? 'expanded' : ''}`}
                  >
                    <div
                      className="accordion-header"
                      onClick={() => toggleCategory(catName)}
                    >
                      <div className="accordion-title-container">
                        <span>{catName}</span>
                        {activeCount > 0 && (
                          <span className="active-count-badge">{activeCount}</span>
                        )}
                      </div>
                      <span className="accordion-arrow">▼</span>
                    </div>

                    {isExpanded && (
                      <div className="accordion-content">
                        {themes.map(theme => (
                          <label
                            key={theme}
                            className={`theme-chip ${selectedThemes[theme] ? 'active' : ''} ${jobStatus === 'generating' ? 'disabled' : ''}`}
                            htmlFor={`checkbox-${theme}`}
                          >
                            <input
                              id={`checkbox-${theme}`}
                              type="checkbox"
                              className="theme-checkbox"
                              checked={selectedThemes[theme]}
                              onChange={() => toggleTheme(theme)}
                              disabled={jobStatus === 'generating'}
                            />
                            <span className="theme-chip-label">{formatThemeName(theme)}</span>
                          </label>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;

