import gymnasium as gym
import metaworld
import numpy as np

from envs.base_env import BaseBenchmarkEnv


class MetaWorldReachEnv(BaseBenchmarkEnv):
    def __init__(self, task_name, render_mode=None):
        super().__init__(task_name=task_name)

        self.env = gym.make("Meta-World/MT1", env_name=task_name, render_mode=render_mode)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        info = self.build_info(obs, info)
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["dense_reward"] = reward
        info = self.build_info(obs, info)
        return obs, reward, terminated, truncated, info 
    def get_coverage_id(self, obs):
        obs = np.asarray(obs)
        pos = obs[:3]
        bins = np.round(pos, 1)
        return tuple(bins.tolist())

    def is_success(self, obs, info=None) -> bool:
        return bool(info.get("success", 0.0)) if info is not None else False

    def close(self):
        self.env.close()

    def get_target_pos(self):
        return np.asarray(self.env.unwrapped._target_pos)