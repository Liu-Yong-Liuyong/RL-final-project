import gymnasium as gym
from envs.base_env import BaseBenchmarkEnv


class SparseCartPoleEnv(BaseBenchmarkEnv):
    def __init__(self, target_x: float = 1.5):
        super().__init__(task_name="SparseCartPole")
        self.env = gym.make("CartPole-v1")
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.target_x = target_x
        self._success_this_episode = False

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._success_this_episode = False
        info["success"] = False
        info = self.build_info(obs, info)
        return obs, info

    def step(self, action):
        obs, _dense_reward, terminated, truncated, info = self.env.step(action)

        cart_pos = float(obs[0])

        # reward only fires once per episode to keep the signal sparse
        if cart_pos >= self.target_x and not terminated and not self._success_this_episode:
            success = True
            sparse_reward = 1.0
            self._success_this_episode = True
        else:
            success = False
            sparse_reward = 0.0

        info["success"] = success
        info["dense_reward"] = _dense_reward
        info = self.build_info(obs, info)

        return obs, sparse_reward, terminated, truncated, info

    def get_coverage_id(self, obs):
        pos_bin = round(float(obs[0]), 1)
        angle_bin = round(float(obs[2]), 1)
        return (pos_bin, angle_bin)

    def is_success(self, obs, info=None):
        if info is not None and "success" in info:
            return bool(info["success"])
        return False

    def close(self):
        self.env.close()
