import numpy as np
import gymnasium as gym
from gymnasium import spaces
from envs.base_env import BaseBenchmarkEnv

class GridWorldEnv(BaseBenchmarkEnv):
    def __init__(self, width=10, height=10, goal_pos=(9, 9)):
        super().__init__(task_name="SparseGridWorld")
        self.width = width
        self.height = height
        self.goal_pos = tuple(goal_pos)
        
        # Observation space: (x, y) coordinates
        self.observation_space = spaces.Box(
            low=np.array([0, 0]), 
            high=np.array([width-1, height-1]), 
            dtype=np.int32
        )
        # Action space: 0=up, 1=right, 2=down, 3=left (aligned with baselines.py design)
        self.action_space = spaces.Discrete(4)
        
        self.current_pos = [0, 0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start from top-left corner by default
        self.current_pos = [0, 0]
        
        obs = np.array(self.current_pos, dtype=np.int32)
        info = {"success": False}
        info = self.build_info(obs, info)
        return obs, info

    def step(self, action):
        x, y = self.current_pos
        
        # Update coordinates based on action (handle boundary collisions)
        if action == 0: y = max(0, y - 1)          # up
        elif action == 1: x = min(self.width - 1, x + 1) # right
        elif action == 2: y = min(self.height - 1, y + 1)# down
        elif action == 3: x = max(0, x - 1)          # left
            
        self.current_pos = [x, y]
        obs = np.array(self.current_pos, dtype=np.int32)
        
        # Sparse reward: only give 1.0 when reaching the goal, otherwise 0.0
        terminated = tuple(self.current_pos) == self.goal_pos
        reward = 1.0 if terminated else 0.0
        truncated = False
        
        info = {"success": terminated}
        info = self.build_info(obs, info)
        
        return obs, reward, terminated, truncated, info

    def get_coverage_id(self, obs):
        # Return tuple (x, y) as the unique ID for coverage calculation
        return tuple(obs.tolist())

    def is_success(self, obs, info=None):
        if info is not None and "success" in info:
            return bool(info["success"])
        return tuple(obs.tolist()) == self.goal_pos
    
    def render(self):
        # Simple print out text version of the map
        for y in range(self.height):
            row_str = ""
            for x in range(self.width):
                if [x, y] == self.current_pos:
                    row_str += "🤖"  # Agent current position
                elif (x, y) == self.goal_pos:
                    row_str += "🏁"  # Goal
                else:
                    row_str += "⬜"  # Empty cell
            print(row_str)
        print("-" * 20)