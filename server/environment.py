import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import Action, Observation, State, Reward

def clamp_score(score: float) -> float:
    """The Nuclear Clamp: Forces everything strictly between 0.05 and 0.95."""
    try:
        val = float(score)
        return round(float(np.clip(val, 0.05, 0.95)), 4)
    except:
        return 0.5000

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
        if task_id == "point_outbreak":
            self.current_state.pest_levels[5][5] = 4.0
        elif task_id == "resource_dilemma":
            self.current_state.pest_levels[0][0] = 5.0
            self.current_state.pest_levels[9][9] = 5.0
        elif task_id == "resistance_test":
            self.current_state.pest_levels[5][5] = 4.0
        return self._get_obs()

    def state(self):
        return self.current_state.dict()

    def step(self, action: Action):
        x, y = action.coordinate
        reward_val = 0.1  # Safe starting reward

        # Tool names FIXED to match your models.Literal exactly
        if action.tool == "scout":
            self.current_state.total_spent += 10
            reward_val = 0.1
        elif action.tool == "apply_neem_oil":
            self.current_state.total_spent += 2
            pest_before = self.current_state.pest_levels[x][y]
            self.current_state.pest_levels[x][y] = max(0.0, pest_before - 2.0)
            reward_val = 0.1 + (pest_before * 0.02)
        elif action.tool == "apply_chemical":
            self.current_state.total_spent += 5
            if not self.current_state.has_resistance:
                self.current_state.pest_levels[x][y] = 0.0
                reward_val = 0.3
            else:
                reward_val = 0.05
        elif action.tool == "biological_control":
            self.current_state.total_spent += 15
            if self.current_state.has_resistance:
                self.current_state.pest_levels[x][y] = 0.0
                reward_val = 0.8
            else:
                reward_val = 0.05
        elif action.tool == "abandon_cell":
            self.current_state.grid_health[x][y] = 0.0
            self.current_state.pest_levels[x][y] = 0.0
            reward_val = 0.05

        self._simulate_growth()
        self.current_state.turns_since_infestation += 1
        
        max_budget = 55.0 if self.task_id == "resource_dilemma" else 100.0
        done = self.current_state.total_spent >= max_budget or np.mean(self.current_state.grid_health) < 0.5
        
        if done:
            reward_val = self._grade_final()
        
        return self._get_obs(), clamp_score(reward_val), done, {"spent": self.current_state.total_spent}

   def _grade_final(self) -> float:
        """
        Final episode score strictly in (0.01, 0.99).
        Ensures the score never collapses to 0.0 even at max budget.
        """
        health_grid = np.array(self.current_state.grid_health)
        pest_grid = np.array(self.current_state.pest_levels)
        max_budget = 55.0 if self.task_id == "resource_dilemma" else 100.0
        
        # 1. Base Health Score (0.1 to 0.9 range)
        avg_health = float(np.mean(health_grid)) / 9.0
        
        # 2. Pest Reduction Score
        avg_pest = float(np.mean(np.clip(pest_grid, 0, 100)))
        pest_reduction = 1.0 - (avg_pest / 100.0)
        
        # 3. Efficiency Floor: prevents exactly 0.0 when budget is spent
        raw_eff = 1.0 - (self.current_state.total_spent / max_budget)
        efficiency = max(0.1234, min(raw_eff, 0.9876)) 

        if self.task_id == "point_outbreak":
            raw = (avg_health * 0.6) + (pest_reduction * 0.4)
        elif self.task_id == "resource_dilemma":
            raw = (avg_health * 0.5) + (pest_reduction * 0.3) + (efficiency * 0.2)
        elif self.task_id == "resistance_test":
            turns = self.current_state.turns_since_infestation
            early_bonus = max(0.05, 0.25 - (turns * 0.03))
            raw = (avg_health * 0.45) + (pest_reduction * 0.30) + early_bonus
        else:
            raw = (avg_health * 0.5) + (pest_reduction * 0.5)
            
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
                    for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        ni, nj = i + di, j + dj
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
            message=f"Status: {self.task_id}"
        )
