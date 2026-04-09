import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import Action, Observation, State, Reward

def clamp_score(score: float) -> float:
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
        reward_val = 0.05  # safe default

        if action.tool == "scout":
            self.current_state.total_spent += 10
            pest_at_cell = self.current_state.pest_levels[x][y]
            reward_val = 0.05 + min(pest_at_cell * 0.01, 0.09)

        elif action.tool in ("neem_oil", "apply_neem_oil"):
            self.current_state.total_spent += 2
            pest_before = self.current_state.pest_levels[x][y]
            self.current_state.pest_levels[x][y] = max(0.0, pest_before - 2.0)
            reduction = pest_before - self.current_state.pest_levels[x][y]
            reward_val = 0.05 + min(reduction * 0.02, 0.09)

        elif action.tool in ("chemical", "apply_chemical"):
            self.current_state.total_spent += 5
            pest_before = self.current_state.pest_levels[x][y]
            if not self.current_state.has_resistance:
                self.current_state.pest_levels[x][y] = 0.0
                reward_val = 0.30
            else:
                reward_val = 0.05

        elif action.tool == "biological_control":
            self.current_state.total_spent += 15
            if self.current_state.has_resistance:
                self.current_state.pest_levels[x][y] = 0.0
                reward_val = 0.75
            else:
                reward_val = 0.05

        elif action.tool == "abandon_cell":
            self.current_state.grid_health[x][y] = 0.0
            self.current_state.pest_levels[x][y] = 0.0
            reward_val = 0.05

        self._simulate_growth()
        self.current_state.turns_since_infestation += 1

        max_budget = 55.0 if self.task_id == "resource_dilemma" else 100.0

        # Budget-only termination — no health check to avoid premature done
        done = self.current_state.total_spent >= max_budget

        if done:
            reward_val = self._grade_final()

        return self._get_obs(), clamp_score(reward_val), done, {
            "spent": self.current_state.total_spent
        }

    def _grade_final(self) -> float:
        health_grid = np.array(self.current_state.grid_health)
        pest_grid = np.array(self.current_state.pest_levels)
        max_budget = 55.0 if self.task_id == "resource_dilemma" else 100.0

        avg_health = float(np.mean(health_grid)) / 9.0
        avg_pest = float(np.mean(np.clip(pest_grid, 0, 100)))
        pest_reduction = 1.0 - (avg_pest / 100.0)

        raw_eff = 1.0 - (self.current_state.total_spent / max_budget)
        efficiency = max(0.15, min(raw_eff, 0.90))

        if self.task_id == "point_outbreak":
            raw = (avg_health * 0.6) + (pest_reduction * 0.4)
        elif self.task_id == "resource_dilemma":
            inner = float(np.mean(health_grid[2:8, 2:8])) / 9.0
            raw = (inner * 0.5) + (pest_reduction * 0.3) + (efficiency * 0.2)
        elif self.task_id == "resistance_test":
            turns = self.current_state.turns_since_infestation
            early_bonus = max(0.05, 0.20 - (turns * 0.02))
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
        self.current_state.pest_levels = np.clip(
            next_pest_grid, 0.0, 100.0
        ).tolist()
        self.current_state.grid_health = np.clip(
            health_grid, 0.0, 9.0
        ).tolist()

    def _get_obs(self):
        max_budget = 55.0 if self.task_id == "resource_dilemma" else 100.0
        heatmap = [[int(h) for h in row] for row in self.current_state.grid_health]
        return Observation(
            heatmap=heatmap,
            sensor_data={"core": self.current_state.pest_levels[5][5]},
            remaining_budget=max_budget - self.current_state.total_spent,
            message=f"Task: {self.task_id} | Budget: {max_budget - self.current_state.total_spent:.1f}"
        )
