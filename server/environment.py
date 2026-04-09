import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import Action, Observation, State

def clamp_score(score: float) -> float:
    return round(float(np.clip(score, 0.0123, 0.9876)), 4)

class AgriGuardEnv:
    def __init__(self):
        self.task_id = "point_outbreak"
        self.reset()

    def reset(self, task_id="point_outbreak"):
        self.task_id = task_id
        self.current_state = State(
            grid_health=[[9.0 for _ in range(10)] for _ in range(10)],
            pest_levels=[[0.0 for _ in range(10)] for _ in range(10)],
            chemical_usage_grid=[[0 for _ in range(10)] for _ in range(10)],
            has_resistance=(task_id == "resistance_test"),
            total_spent=0.0,
            total_chemical_count=0,
            turns_since_infestation=0
        )
        # Task Initializations
        if task_id == "point_outbreak":
            self.current_state.pest_levels[5][5] = 4.0
        elif task_id == "resource_dilemma":
            self.current_state.pest_levels[0][0] = 5.0
            self.current_state.pest_levels[9][9] = 5.0
        return self._get_obs()

    def step(self, action: Action):
        # 1. Validation Firewall
        if not isinstance(action.coordinate, (list, tuple)) or len(action.coordinate) != 2:
            return self._get_obs(), 0.1234, False, {"error": "Invalid coords"}
        
        x, y = action.coordinate
        if not (0 <= x < 10 and 0 <= y < 10):
            return self._get_obs(), 0.1234, False, {"error": "Out of bounds"}

        reward_val = 0.1  # Baseline
        
        # 2. Tool Logic with Sustainability & Resistance Penalties
        if action.tool == "scout":
            self.current_state.total_spent += 10.0
            reward_val = 0.15 + (self.current_state.pest_levels[x][y] * 0.02)

        elif action.tool in ("chemical", "apply_chemical"):
            self.current_state.total_spent += 5.0
            self.current_state.total_chemical_count += 1
            self.current_state.chemical_usage_grid[x][y] += 1
            
            # Soil Toxicity: Permanent small health hit
            self.current_state.grid_health[x][y] = max(0.0, self.current_state.grid_health[x][y] - 0.3)
            
            # Emergent Resistance: Spamming chemicals makes them fail
            if self.current_state.chemical_usage_grid[x][y] >= 3:
                self.current_state.has_resistance = True
            
            if not self.current_state.has_resistance:
                self.current_state.pest_levels[x][y] = 0.0
                reward_val = 0.4
            else:
                reward_val = 0.05

        elif action.tool == "biological_control":
            self.current_state.total_spent += 15.0
            if self.current_state.has_resistance:
                self.current_state.pest_levels[x][y] = 0.0
                reward_val = 0.8
            else:
                reward_val = 0.2

        elif action.tool == "abandon_cell":
            self.current_state.grid_health[x][y] = 0.0
            self.current_state.pest_levels[x][y] = 0.0
            reward_val = 0.05

        # 3. Simulation: Growth & Spatial Spread
        self._simulate_growth_and_spread()
        self.current_state.turns_since_infestation += 1

        # 4. Budget & Grade
        max_budget = 55.0 if self.task_id == "resource_dilemma" else 100.0
        done = self.current_state.total_spent >= max_budget
        
        if done:
            reward_val = self._grade_final()

        return self._get_obs(), clamp_score(reward_val), done, {"spent": self.current_state.total_spent}

    def _simulate_growth_and_spread(self):
        pests = np.array(self.current_state.pest_levels)
        health = np.array(self.current_state.grid_health)
        next_pests = np.copy(pests)

        for i in range(10):
            for j in range(10):
                if pests[i][j] > 1.0 and health[i][j] > 0:
                    health[i][j] = max(0.0, health[i][j] - (pests[i][j] * 0.05))
                    # Spatial Spread: Pests move to neighbors
                    for di, dj in [(0,1),(0,-1),(1,0),(-1,0)]:
                        ni, nj = i+di, j+dj
                        if 0 <= ni < 10 and 0 <= nj < 10:
                            next_pests[ni][nj] += 0.2
        
        self.current_state.pest_levels = np.clip(next_pests, 0.0, 10.0).tolist()
        self.current_state.grid_health = health.tolist()

    def _grade_final(self) -> float:
        # Multi-Objective Grading
        avg_health = np.mean(self.current_state.grid_health) / 9.0
        max_b = 55.0 if self.task_id == "resource_dilemma" else 100.0
        efficiency = max(0.0, (max_b - self.current_state.total_spent) / max_b)
        # Penalize if chemicals > 25% of budget
        sustainability = max(0.0, 1.0 - (self.current_state.total_chemical_count / 12.0))
        
        score = (0.5 * avg_health) + (0.2 * efficiency) + (0.3 * (sustainability**1.5))
        return score

    def _get_obs(self):
        heatmap = [[int(h) for h in row] for row in self.current_state.grid_health]
        return Observation(
            heatmap=heatmap,
            sensor_data={"core_pest": self.current_state.pest_levels[5][5]},
            remaining_budget=100.0 - self.current_state.total_spent,
            message=f"Task: {self.task_id} | Health: {np.mean(self.current_state.grid_health):.1f}/9.0"
        )
