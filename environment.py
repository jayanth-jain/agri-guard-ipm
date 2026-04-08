import numpy as np
import sys
import os

# Robust path handling for models import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import Action, Observation, State, Reward

class AgriGuardEnv:
    def __init__(self):
        self.task_id = "point_outbreak" 
        # We define self.current_state to avoid naming conflicts with the method
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
        
        if task_id == "point_outbreak":
            self.current_state.pest_levels[5][5] = 4.0
        elif task_id == "resource_dilemma":
            self.current_state.pest_levels[0][0] = 5.0
            self.current_state.pest_levels[9][9] = 5.0
        
        return self._get_obs()

    # --- ADD THIS METHOD FOR OPENENV COMPLIANCE ---
    def state(self):
        """Returns the absolute ground-truth state as a dictionary."""
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
        reward_val = 0.0
        
        if action.tool == "scout":
            self.current_state.total_spent += 10
        elif action.tool == "apply_neem_oil":
            self.current_state.total_spent += 2
            self.current_state.pest_levels[x][y] = max(0, self.current_state.pest_levels[x][y] - 2.0)
        elif action.tool == "apply_chemical":
            self.current_state.total_spent += 5
            if not self.current_state.has_resistance:
                self.current_state.pest_levels[x][y] = 0.0
        elif action.tool == "biological_control":
            self.current_state.total_spent += 15
            if self.current_state.has_resistance:
                self.current_state.pest_levels[x][y] = 0.0
                decay_map = {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.5, 4: 0.2}
                reward_val = decay_map.get(self.current_state.turns_since_infestation, 0.05)
        elif action.tool == "abandon_cell":
            self.current_state.grid_health[x][y] = 0.0
            self.current_state.pest_levels[x][y] = 0.0

        self._simulate_growth()
        self.current_state.turns_since_infestation += 1
        
        max_budget = 55.0 if self.task_id == "resource_dilemma" else 100.0
        done = self.current_state.total_spent >= max_budget or np.mean(self.current_state.grid_health) < 0.5
        
        return self._get_obs(), reward_val, done, {"spent": self.current_state.total_spent}

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
            message=f"Task: {self.task_id} | Budget: {max_budget}"
        )