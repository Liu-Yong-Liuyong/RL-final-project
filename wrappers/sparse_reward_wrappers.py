#for carpole and robot arms...
import gymnasium as gym


class SparseRewardWrapper(gym.Wrapper):
    def __init__(self, env, success_reward=10.0, failure_reward=0.0):
        super().__init__(env)
        self.success_reward = success_reward
        self.failure_reward = failure_reward

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        #sparse_reward = self.success_reward if info.get("success", False) else self.failure_reward
        sparse_reward = self.success_reward if info.get("success", False) else 0.01 * reward
        info["original_reward"] = reward

        return obs, sparse_reward, terminated, truncated, info