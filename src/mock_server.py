from ChessGeneration import Position, Evaluation
import uuid
import time
import random
from enum import StrEnum
from dataclasses import dataclass, replace
from typing import List, Dict, Any, Optional
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

class JobStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class GenerationJob:
    id: str
    status: JobStatus
    created_at: float
    rating: int
    themes: List[str]
    puzzle: Optional[Evaluation] = None

# In-memory job storage
jobs: Dict[str, GenerationJob] = {}  # Use a queue in the real product to take the jobs in order

# List of mock evaluations to return upon completion
MOCK_EVALUATIONS = [
    Evaluation(
        position=Position(
            fen="3r3r/pR1nkp2/4p1p1/P1P5/8/2n5/5PPQ/5RK1 b - - 0 30",
            move="-",
            base_rating=-1.0,
            base_themes=["long", "middlegame", "mate", "attraction", "anastasiaMate"]
        ),
        legal=True,
        unique_solution=True,
        counter_intuitive_solution=False,
        counter_intuitive_value=0.0,
        themes_match=True,
        mainline=["Ne2+", "Kh1", "Rxh2+", "Kxh2", "Rh8#"]
    ),
    Evaluation(
        position=Position(
            fen="2r1n1k1/p4ppp/1p2p3/6q1/8/3QP2P/4BPPK/R3N3 b - - 2 39",
            move="-",
            base_rating=-1.0,
            base_themes=["middlegame", "advantage", "short", "fork"]
        ),
        legal=True,
        unique_solution=True,
        counter_intuitive_solution=False,
        counter_intuitive_value=0.0,
        themes_match=True,
        mainline=["Qe5+", "f4", "Qxa1"]
    )
]

class GenerateRequest(BaseModel):
    rating: int
    themes: List[str]

class JobStatusResponse(BaseModel):
    status: JobStatus
    puzzle: Optional[Evaluation] = None

class GenerateResponse(BaseModel):
    jobId: str

class CancelResponse(BaseModel):
    status: JobStatus

@app.post("/api/puzzles/generate", response_model=GenerateResponse)
async def generate_puzzle(request: GenerateRequest):
    job_id = str(uuid.uuid4())
    jobs[job_id] = GenerationJob(
        id=job_id,
        status=JobStatus.PENDING,
        created_at=time.time(),
        rating=request.rating,
        themes=request.themes
    )
    print(f"[Mock Server] Created generation job {job_id}: rating={request.rating}, themes={request.themes}")
    return GenerateResponse(jobId=job_id)

@app.get("/api/puzzles/status/{job_id}", response_model=JobStatusResponse)
async def check_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = jobs[job_id]
    
    if job.status == JobStatus.PENDING:
        mock_eval = random.choice(MOCK_EVALUATIONS)
        
        is_unique = random.random() < 0.8  # 80% chance of unique solution
        counter_value = round(random.uniform(-0.05, 2.5), 2)
        is_counter_intuitive = counter_value > 0.1
        
        job.status = JobStatus.COMPLETED
        job.puzzle = replace(
            mock_eval,
            unique_solution=is_unique,
            counter_intuitive_solution=is_counter_intuitive,
            counter_intuitive_value=counter_value
        )
        print(f"[Mock Server] Job {job_id} marked COMPLETED")
            
    return JobStatusResponse(
        status=job.status,
        puzzle=job.puzzle
    )

@app.post("/api/puzzles/cancel/{job_id}", response_model=CancelResponse)
async def cancel_puzzle(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = jobs[job_id]
    if job.status == JobStatus.PENDING:
        job.status = JobStatus.CANCELLED
        print(f"[Mock Server] Job {job_id} CANCELLED by user request")
        return CancelResponse(status=JobStatus.CANCELLED)
    else:
        print(f"[Mock Server] Cancel requested for job {job_id} but status is already {job.status}")
        return CancelResponse(status=job.status)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
