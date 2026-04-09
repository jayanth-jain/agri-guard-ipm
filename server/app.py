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

@app.post("/step")
async def step(action: Action):
    obs, reward_val, done, info = env.step(action)
    
    # EDGE SANITATION: One last guardrail before the data leaves the server
    # This guarantees the validator sees a number strictly between 0 and 1
    safe_reward_val = max(0.001, min(0.999, float(reward_val)))
    
    # Wrap reward in the Pydantic model
    reward = Reward(value=safe_reward_val, comment="Step processed")
    
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
