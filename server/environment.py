import numpy as np
import sys
import os

# Robust path handling for models import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import Action, Observation, State, Reward

def clamp_score(score: float) -> float:
    """
    STRICT CONSTRAINT: Ensures reward is never exactly 0.0 or 1.0.
    Returns values between 0.001 and 0.999.
    """
    try:
        val = float(score)
        return round(float(np.clip(val, 0.001, 0.999)), 4)
    except:
        return 0.5

class AgriGuardEnv:
    def __init__(self):
        self.task_id = "point_outbreak"
        self.current_state = None
        self.reset()

    def reset(self, task_id="point_outbreak"):
        self.task_id = task_id
        self.current_state = State(
            grid_health=[[9.0 for _ in range(10)] for _ in range(10)],
            pest_levels=[[0.0 for _ in range(10)] for _ in range(10)],
            has_resistance=(task_id == "resistance_test"),
            turns_since_infestation=0,
            total_spent=0.0
        )
        
        # Initialize pests based on task
        if task_id == "point_outbreak":
            self.current_state.pest_levels[5][5] = 4.0
        elif task_id == "resource_dilemma":
            self.current_state.pest_levels[0][0] = 5.0
            self.current_state.pest_levels[9][9] = 5.0
        elif task_id == "resistance_test":
            self.current_state.pest_levels[5][5] = 4.0
        
        return self._get_obs()

    def state(self):
        """Standard OpenEnv ground-truth retrieval."""
        return {
            "task_id": self.task_id,
            "grid_health": self.current_state.grid_health,
            "pest_levels": self.current_state.pest_levels,
            "has_resistance": self.current_state.has_resistance,
            "total_spent": self.current_state.total_spent,
            "turns": self.current_state.turns_since_infestation
        }

    def step(self, action: Action):
        x, y = action.coordinate
        # Default intermediate reward (must not be 0.0)
        reward_val = 0.01 
        
        # Tool Logic
        if action.tool == "scout":
            self.current_state.total_spent += 10
            reward_val = 0.05
        elif action.tool == "apply_neem_oil":
            self.current_state.total_spent += 2
            pest_before = self.current_state.pest_levels[x][y]
            self.current_state.pest_levels[x][y] = max(0, self.current_state.pest_levels[x][y] - 2.0)
            reduction = pest_before - self.current_state.pest_levels[x][y]
            reward_val = 0.05 + (reduction * 0.02)
        elif action.tool == "apply_chemical":
            self.current_state.total_spent += 5
            pest_before = self.current_state.pest_levels[x][y]
            if not self.current_state.has_resistance:
                self.current_state.pest_levels[x][y] = 0.0
                reward_val = 0.10 + (pest_before * 0.05)
            else:
                reward_val = 0.002 # Resistance penalty
        elif action.tool == "biological_control":
            self.current_state.total_spent += 15
            if self.current_state.has_resistance:
                self.current_state.pest_levels[x][y] = 0.0
                # Early intervention weighted higher
                decay_map = {0: 0.9, 1: 0.8, 2: 0.7, 3: 0.5, 4: 0.3}
                reward_val = decay_map.get(self.current_state.turns_since_infestation, 0.1)
            else:
                reward_val = 0.005
        elif action.tool == "abandon_cell":
            self.current_state.grid_health[x][y] = 0.0
            self.current_state.pest_levels[x][y] = 0.0
            reward_val = 0.002

        self._simulate_growth()
        self.current_state.turns_since_infestation += 1
        
        # Episode Termination Logic
        max_budget = 55.0 if self.task_id == "resource_dilemma" else 100.0
        done = self.current_state.total_spent >= max_budget or np.mean(self.current_state.grid_health) < 0.5
        
        # If the episode is over, calculate the final task score
        if done:
            reward_val = self._grade_final()
        
        return self._get_obs(), clamp_score(reward_val), done, {"spent": self.current_state.total_spent}

    def _grade_final(self) -> float:
        """Weighted grader for the entire task."""
        avg_health = np.mean(self.current_state.grid_health) / 9.0
        pest_reduction = 1.0 - (np.mean(self.current_state.pest_levels) / 100.0)
        
        if self.task_id == "point_outbreak":
            raw = (avg_health * 0.6) + (pest_reduction * 0.4)
        else:
            max_budget = 55.0 if self.task_id == "resource_dilemma" else 100.0
            efficiency = 1.0 - (self.current_state.total_spent / max_budget)
            raw = (avg_health * 0.4) + (efficiency * 0.6)
            
        return clamp_score(raw)

    def _simulate_growth(self):
        pest_grid = np.array(self.current_state.pest_levels)
        health_grid = np.array(self.current_state.grid_health)
        next_pest_grid = np.copy(pest_grid)
        
        for i in range(10):
            for j in range(10):
                if pest_grid[i][j] > 1.0 and health_grid[i][j] > 0:
                    damage = pest_grid[i][j] * 0.05
                    health_grid[i][j] = max(0.0, health_grid[i][j] - damage)
                    for di, dj in [(0,1), (0,-1), (1,0), (-1,0)]:
                        ni, nj = i+di, j+dj
                        if 0 <= ni < 10 and 0 <= nj < 10:
                            next_pest_grid[ni][nj] += 0.3
                            
        self.current_state.pest_levels = np.clip(next_pest_grid, 0.0, 100.0).tolist()
        self.current_state.grid_health = np.clip(health_grid, 0.0, 9.0).tolist()

    def _get_obs(self):
        max_budget = 55.0 if self.task_id == "resource_dilemma" else 100.0
        heatmap = [[int(h) for h in row] for row in self.current_state.grid_health]
        return Observation(
            heatmap=heatmap,
            sensor_data={"core": self.current_state.pest_levels[5][5]},
            remaining_budget=max_budget - self.current_state.total_spent,
            message=f"Task: {self.task_id} | Budget: {max_budget - self.current_state.total_spent:.1f}"
        )
