from ChessGeneration import Position
import uuid
import time
import random
from typing import List, Dict, Any
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Chess Puzzle Generator Mock API")

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job storage
jobs: Dict[str, Dict[str, Any]] = {}

# List of mock positions to return upon completion
MOCK_POSITIONS = [
    {
        "fen": "3r3r/pR1nkp2/4p1p1/P1P5/8/2n5/5PPQ/5RK1 b - - 0 30",
        "move": "c3e2",
        "base_rating": 2000.0,
        "base_themes": ["long", "middlegame", "mate", "attraction", "anastasiaMate"],
        "mainline": ["Ne2+", "Kh1", "Rxh2+", "Kxh2", "Rh8#"]
    },
    {
        "fen": "2r1n1k1/p4ppp/1p2p3/6q1/8/3QP2P/4BPPK/R3N3 b - - 2 39",
        "move": "-",
        "base_rating": 1000.0,
        "base_themes": ["middlegame", "advantage", "short", "fork"],
        "mainline": ["Qe5+", "f4", "Qxa1"]
    },
]

class GenerateRequest(BaseModel):
    rating: int
    themes: List[str]

@app.post("/api/puzzles/generate")
async def generate_puzzle(request: GenerateRequest):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "id": job_id,
        "status": "pending",
        "created_at": time.time(),
        "rating": request.rating,
        "themes": request.themes
    }
    print(f"[Mock Server] Created generation job {job_id}: rating={request.rating}, themes={request.themes}")
    return {"jobId": job_id}

@app.get("/api/puzzles/status/{job_id}")
async def check_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = jobs[job_id]
    
    if job["status"] == "pending":
        elapsed = time.time() - job["created_at"]
        if elapsed >= 5.0:
            mock_pos = random.choice(MOCK_POSITIONS)
            
            is_unique = random.random() < 0.8  # 80% chance of unique solution
            is_counter_intuitive = random.random() < 0.4  # 40% chance of counter-intuitive moves
            
            job["status"] = "completed"
            job["puzzle"] = {
                "position": {
                    "fen": mock_pos["fen"],
                    "move": mock_pos["move"],
                    "base_rating": float(job["rating"]),
                    "base_themes": job["themes"]
                },
                "legal": True,
                "unique_solution": is_unique,
                "counter_intuitive_solution": is_counter_intuitive,
                "counter_intuitive_value": round(random.uniform(0.5, 2.5), 2) if is_counter_intuitive else 0.0,
                "themes_match": True,
                "mainline": mock_pos["mainline"]
            }
            print(f"[Mock Server] Job {job_id} marked COMPLETED")
            
    return {
        "status": job["status"],
        "puzzle": job.get("puzzle")
    }

@app.post("/api/puzzles/cancel/{job_id}")
async def cancel_puzzle(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = jobs[job_id]
    if job["status"] == "pending":
        job["status"] = "cancelled"
        print(f"[Mock Server] Job {job_id} CANCELLED by user request")
        return {"status": "cancelled"}
    else:
        print(f"[Mock Server] Cancel requested for job {job_id} but status is already {job['status']}")
        return {"status": job["status"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
