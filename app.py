from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from models import Action, Observation, Reward
from .environment import AgriGuardEnv

app = FastAPI(title="Agri-Guard API")
env = AgriGuardEnv()

@app.get("/")
async def root():
    # This ensures no one sees a blank page; it redirects to the UI
    return RedirectResponse(url="/docs")

@app.post("/reset")
async def reset(task_id: str = "point_outbreak"):
    return env.reset(task_id)

@app.post("/step")
async def step(action: Action):
    obs, reward_val, done, info = env.step(action)
    # Wrap reward in the Pydantic model
    reward = Reward(value=reward_val, comment="Step processed")
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
async def get_state():
    """Returns the absolute ground-truth state of the environment."""
    return env.state()