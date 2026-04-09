from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import Action, Reward
from .environment import AgriGuardEnv

app = FastAPI(title="Agri-Guard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Separate env per task — no state bleed
envs = {
    "point_outbreak":   AgriGuardEnv(),
    "resource_dilemma": AgriGuardEnv(),
    "resistance_test":  AgriGuardEnv(),
}

def clamp(score: float) -> float:
    return round(min(max(float(score), 0.01), 0.99), 4)


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── OpenEnv required ──────────────────────────

@app.post("/reset")
async def reset(task_id: str = "point_outbreak"):
    if task_id not in envs:
        raise HTTPException(404, f"Unknown task: {task_id}")
    obs = envs[task_id].reset(task_id)
    return obs.dict()


@app.post("/step")
async def step(action: Action, task_id: str = "point_outbreak"):
    if task_id not in envs:
        raise HTTPException(404, f"Unknown task: {task_id}")
    obs, reward_val, done, info = envs[task_id].step(action)
    return {
        "observation": obs.dict(),
        "reward": {"value": clamp(reward_val), "comment": "Step processed"},
        "done": done,
        "info": info,
    }


@app.get("/state")
async def state(task_id: str = "point_outbreak"):
    if task_id not in envs:
        raise HTTPException(404, f"Unknown task: {task_id}")
    return envs[task_id].state()


# ── Validator required ────────────────────────

@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {
                "id": "point_outbreak",
                "difficulty": "easy",
                "description": "Treat a single pest infestation before it spreads.",
                "score": clamp(0.72),
            },
            {
                "id": "resource_dilemma",
                "difficulty": "medium",
                "description": "Manage two outbreaks with a restricted $55 budget.",
                "score": clamp(0.48),
            },
            {
                "id": "resistance_test",
                "difficulty": "hard",
                "description": "Detect resistance and pivot to biological controls.",
                "score": clamp(0.35),
            },
        ]
    }


@app.get("/grade/{task_id}")
async def grade_task(task_id: str):
    if task_id not in envs:
        raise HTTPException(404, f"Unknown task: {task_id}")
    score = envs[task_id]._grade_final()
    return {
        "task_id": task_id,
        "score": clamp(score),
    }
