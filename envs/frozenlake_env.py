"""
FrozenLake benchmark adapter.
"""

import gymnasium as gym
from envs.base_env import BaseBenchmarkEnv


class FrozenLakeEnv(BaseBenchmarkEnv):
    def __init__(self, map_name="4x4", is_slippery=True):
        super().__init__(task_name="FrozenLake")
        self.env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, seed=None, options=None):
      obs, info = self.env.reset(seed=seed, options=options)
      info["success"] = False
      info = self.build_info(obs, info)
      return obs, info

    def step(self, action):
      obs, reward, terminated, truncated, info = self.env.step(action)
      info["success"] = bool(reward > 0)
      info = self.build_info(obs, info)
      return obs, reward, terminated, truncated, info

    def get_coverage_id(self, obs):
        # FrozenLake observation is usually an integer state index
        return int(obs)

    def is_success(self, obs, info=None):
        # In FrozenLake, reaching the goal gives reward 1
        # But reward is not passed directly here, so we infer success from state if needed
        # For simplicity, we use info if already present, otherwise False
        if info is not None and "success" in info:
            return bool(info["success"])
        return False
