import numpy as np
import os

from enum import Enum
from itertools import product as cartesian
from collections import namedtuple
from typing import *


class Action(Enum):
    LEFT  = (-1, 0)
    RIGHT = (1, 0)
    UP    = (0, 1)
    DOWN  = (0, -1)

    @property
    def delta(self):
        return self.value

class GridCell(Enum):
    OPEN  = 0
    WALL  = 1
    GOAL  = 2
    START = 9
    
    @property
    def reward(self):
        if self is GridCell.OPEN : return -.01
        if self is GridCell.WALL : return -.5
        if self is GridCell.GOAL : return 1.0
        if self is GridCell.START: return -.1

StepResult = namedtuple("StepResult",["reward","nextState","isGoal"])

class Grid:
    def __init__(self,file_path:str):
        self.rows = None
        self.cols = None
        self.grid = None
        self.tag  = None
        self.file_path = file_path 
        if os.path.exists(file_path): 
            self.read(file_path)
            self.file_loaded = True
        else:
            print(f"[ERROR] File Not Found: {file_path}")
            self.file_loaded = False	
    
    def read(self, file_path):
        base, ext = os.path.basename(file_path).split(".")
        self.tag  = base 
        if ext == "maze":
            with open(file_path, "rb") as f:
                rows = int.from_bytes(f.read(8), 'little')
                cols = int.from_bytes(f.read(8), 'little')
                data = f.read()

            arr = np.frombuffer(data, dtype=np.uint8)
            self.grid = arr.reshape(rows, cols)
            self.rows, self.cols = rows, cols

        elif ext == "npy":
            self.grid = np.load(file_path)
            self.rows, self.cols = self.grid.shape
	
    def debug(self):
        print()
        print("NAME: ",self.tag)
        print("ROWS: ",self.rows)
        print("COLS: ",self.cols)
        print()
        for (r,c) in cartesian(range(self.rows),range(self.cols)):
          print(self.grid[r,c],
                  end="\n" if c == self.cols-1 else "\t")

class MazeEnv(Grid):
    def __init__(self,maze_file_path:str,rewards_scaled=False, pass_through_walls: bool = False,
                 reward_shaping: bool = False, shaping_gamma: float = 0.99):
        """If pass_through_walls is True the agent may move into cells that are walls.
        Behavior:
         - Movement into WALL cells is allowed (unless out-of-bounds).
         - WALL cells, when passed-through, are treated as OPEN for reward calculation.
         - Leaving the grid (out-of-bounds) is still treated as a WALL and blocked.

        If reward_shaping is True, applies potential-based shaping (Ng et al. 1999):
            F(s, s') = shaping_gamma * Phi(s') - Phi(s)
        with Phi(s) = -manhattan(s, goal) / max_dist  in [-1, 0].
        Preserves the optimal policy while providing a dense directional signal.
        """
        super().__init__(maze_file_path)
        self.rewards_scaled     = rewards_scaled
        self.pass_through_walls = pass_through_walls
        self.reward_shaping     = reward_shaping
        self.shaping_gamma      = float(shaping_gamma)

        if self.file_loaded:
            coords = np.where(self.grid == GridCell.START.value)
            self.agent_start = (int(coords[0][0]), int(coords[1][0]))

            coords = np.where(self.grid == GridCell.GOAL.value)
            self.agent_goal = (int(coords[0][0]), int(coords[1][0]))

            self.walls_count = np.sum(self.grid == GridCell.WALL.value)
            self.opens_count = np.sum(self.grid == GridCell.OPEN.value)

            # Manhattan-distance bound used to normalize the potential into [-1, 0].
            self._max_manhattan = max(1, (self.rows - 1) + (self.cols - 1))
    
    def debug(self):
        super().debug()
        print()
        print("agent_start: ",self.agent_start)
        print("agent_goal: ",self.agent_goal)
        print("walls_count: ",self.walls_count)
        print("opens_count: ",self.opens_count)
        print()
    
    def getCellReward(self,cell_t:GridCell):
        scaled_goal = self.opens_count if self.rewards_scaled else 1.0
        scaled_wall = self.walls_count if self.rewards_scaled else 1.0
        if cell_t is GridCell.GOAL:
            return scaled_goal*cell_t.reward
        elif cell_t is GridCell.WALL:
            return scaled_wall*cell_t.reward
        else:
            return cell_t.reward
    
    def potential(self, state: Tuple[int,int]) -> float:
        """Phi(s) = -manhattan(s, goal) / max_dist  in [-1, 0].
        Goal has potential 0; cells far from goal have potential -1.
        Returns 0.0 when reward_shaping is disabled (no-op for callers)."""
        if not self.reward_shaping:
            return 0.0
        dr = abs(state[0] - self.agent_goal[0])
        dc = abs(state[1] - self.agent_goal[1])
        return -float(dr + dc) / float(self._max_manhattan)

    def shaping_bonus(self, state: Tuple[int,int], next_state: Tuple[int,int]) -> float:
        """F(s, s') = gamma * Phi(s') - Phi(s). Zero when shaping is off."""
        if not self.reward_shaping:
            return 0.0
        return self.shaping_gamma * self.potential(next_state) - self.potential(state)

    def getStateGridCell(self,state:Tuple[int,int]) -> GridCell:
        out = (
            state[0] < 0 or
            state[0] >= self.rows or
            state[1] < 0 or
            state[1] >= self.cols
        )
        if out: return GridCell.WALL
        return GridCell(self.grid[state[0],state[1]])
    
    def step(self, state: Tuple[int,int], action: Action) -> StepResult:
        delta = action.delta
        next_state = (state[0] + delta[1], state[1] + delta[0])
        
        out = (
            next_state[0] < 0 or
            next_state[0] >= self.rows or
            next_state[1] < 0 or
            next_state[1] >= self.cols
        )

        if out:
            # Out of bounds -> treat as wall and block movement regardless of pass_through_walls
            cell_t = GridCell.WALL
            reward = self.getCellReward(cell_t)
            is_goal = False
            next_state_final = state
        else:
            cell_value = self.grid[next_state[0], next_state[1]]
            cell_t = GridCell(cell_value)
            is_goal = (cell_t is GridCell.GOAL)

            if cell_t is GridCell.WALL:
                reward = self.getCellReward(GridCell.WALL)
                if self.pass_through_walls:
                    next_state_final = next_state
                else:
                    next_state_final = state
            else:
                reward = self.getCellReward(cell_t)
                next_state_final = next_state

        # Potential-based shaping: gamma * Phi(s') - Phi(s).
        # Adds a directional signal toward the goal without changing the optimal policy.
        if self.reward_shaping:
            reward += self.shaping_bonus(state, next_state_final)

        return StepResult(reward=reward, nextState=next_state_final, isGoal=is_goal)