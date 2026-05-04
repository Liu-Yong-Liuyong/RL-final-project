import gymnasium as gym
import numpy as np
from envs.base_env import BaseBenchmarkEnv


class SparseCartPoleEnv(BaseBenchmarkEnv):
    """
    CartPole-v1 with sparse rewards.

    Dense reward (step survived) is replaced by a single sparse signal:
    +1.0 when the episode is truncated (pole stayed up for the full time limit),
    0.0 for every other step.

    Success = episode ends via truncation (survived), not termination (pole fell).
    Coverage = (angle_bin, angular_velocity_bin) rounded to 1 decimal place.
    """

    def __init__(self):
        super().__init__(task_name="SparseCartPole")
        self.env = gym.make("CartPole-v1")
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        info["success"] = False
        info = self.build_info(obs, info)
        return obs, info

    def step(self, action):
        obs, _dense_reward, terminated, truncated, info = self.env.step(action)

        # sparse: reward only when the agent survives the full episode
        success = truncated and not terminated
        sparse_reward = 1.0 if success else 0.0

        info["success"] = success
        info["dense_reward"] = _dense_reward
        info = self.build_info(obs, info)

        return obs, sparse_reward, terminated, truncated, info

    def get_coverage_id(self, obs):
        # obs = [cart_pos, cart_vel, pole_angle, pole_angular_vel]
        angle_bin = round(float(obs[2]), 1)
        angvel_bin = round(float(obs[3]), 1)
        return (angle_bin, angvel_bin)

    def is_success(self, obs, info=None):
        if info is not None and "success" in info:
            return bool(info["success"])
        return False

    def close(self):
        self.env.close()
