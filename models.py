from pydantic import BaseModel
from typing import List, Dict, Optional

class Action(BaseModel):
    """
    Standard Action model. 
    tool: scout, neem_oil, apply_chemical, biological_control, abandon_cell
    coordinate: [x, y]
    """
    tool: str
    coordinate: List[int]

class Observation(BaseModel):
    """
    Observation schema for the agent.
    Heatmap: 10x10 grid of health (0-9)
    Sensor_data: Localized pest readings
    """
    heatmap: List[List[int]]
    sensor_data: Dict[str, float]
    remaining_budget: float
    message: str

class Reward(BaseModel):
    """
    Used for schema documentation. 
    Actual API returns a flat float for validator compatibility.
    """
    value: float
    comment: Optional[str] = None

class State(BaseModel):
    """
    Internal state tracking for the 'Judgmental' environment.
    """
    grid_health: List[List[float]]
    pest_levels: List[List[float]]
    
    # --- TOP 100 Logic Fields ---
    chemical_usage_grid: List[List[int]]  # Tracks per-cell chemical spam to trigger resistance
    total_chemical_count: int             # Used to calculate the Sustainability Score
    # ----------------------------
    
    has_resistance: bool
    total_spent: float
    turns_since_infestation: int
