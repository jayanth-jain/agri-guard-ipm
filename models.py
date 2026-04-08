from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal

# --- ADD THIS CLASS ---
class Reward(BaseModel):
    value: float = Field(..., description="The reward value between 0.0 and 1.0")
    comment: str = Field(..., description="Reasoning for the reward score")

class Action(BaseModel):
    tool: Literal["scout", "apply_neem_oil", "apply_chemical", "biological_control", "abandon_cell"] = Field(
        ..., description="The agricultural tool to use."
    )
    coordinate: List[int] = Field(..., description="[x, y] coordinates on the 10x10 grid.")

class Observation(BaseModel):
    heatmap: List[List[int]] = Field(..., description="10x10 health grid (0=dead, 9=healthy).")
    sensor_data: Dict[str, float] = Field(..., description="Exact pest counts from 4 corner sensors.")
    scout_report: Optional[Dict[str, Any]] = Field(None, description="Detailed info if scout tool was used.")
    remaining_budget: float
    message: str

class State(BaseModel):
    grid_health: List[List[float]]
    pest_levels: List[List[float]]
    has_resistance: bool
    turns_since_infestation: int = 0
    total_spent: float = 0.0