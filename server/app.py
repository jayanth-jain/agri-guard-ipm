from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import Action, Observation
from .environment import AgriGuardEnv

app = FastAPI(title="Agri-Guard IPM Benchmark")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

envs = {
    "point_outbreak": AgriGuardEnv(),
    "resource_dilemma": AgriGuardEnv(),
    "resistance_test": AgriGuardEnv(),
}

@app.get("/")
async def root(): return RedirectResponse(url="/docs")

@app.post("/reset", response_model=Observation)
async def reset(task_id: str = "point_outbreak"):
    if task_id not in envs: raise HTTPException(404)
    return envs[task_id].reset(task_id)

@app.post("/step")
async def step(action: Action, task_id: str = "point_outbreak"):
    if task_id not in envs: raise HTTPException(404)
    obs, reward, done, info = envs[task_id].step(action)
    return {
        "observation": obs.dict(),
        "reward": float(reward), # Flat Float
        "done": bool(done),
        "info": info
    }

@app.get("/grade/{task_id}")
async def grade(task_id: str):
    if task_id not in envs: raise HTTPException(404)
    score = envs[task_id]._grade_final()
    return {"task_id": task_id, "score": float(score)}

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
