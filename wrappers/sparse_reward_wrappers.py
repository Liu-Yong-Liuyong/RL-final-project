#for carpole and robot arms...
import gymnasium as gym
import numpy as np


class SparseRewardWrapper(gym.Wrapper):
    def __init__(self, env, success_reward=10.0, failure_reward=0.0, dense_scale=0.0):
        super().__init__(env)
        self.success_reward = success_reward
        self.failure_reward = failure_reward
        self.dense_scale = dense_scale

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        #sparse_reward = self.success_reward if info.get("success", False) else self.failure_reward
        # sparse_reward = self.success_reward if info.get("success", False) else 0.01 * reward
        if info.get("success", False):
            sparse_reward = self.success_reward
        else:
            sparse_reward = self.failure_reward + (self.dense_scale * reward)
        info["original_reward"] = reward

        return obs, sparse_reward, terminated, truncated, info



# good at normal ppo, icm, rnd for reach-v3
import gymnasium as gym
import numpy as np


class GoalThresholdRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        success_reward=50.0,
        threshold_rewards=None,
        failure_reward=0.0,
        use_dense_fallback=False,
        dense_coef=0.0,
    ):
        super().__init__(env)
        self.success_reward = success_reward
        self.failure_reward = failure_reward
        self.use_dense_fallback = use_dense_fallback
        self.dense_coef = dense_coef

        # 每個元素: (distance_threshold, reward)
        if threshold_rewards is None:
            threshold_rewards = [
                (0.12, 0.05),
                (0.07, 0.1),
                (0.03, 0.5),
            ]

        # 由大到小排序，方便依序判斷
        self.threshold_rewards = sorted(threshold_rewards, key=lambda x: x[0], reverse=True)

        # 記錄本 episode 已經領過哪些 threshold reward
        self.threshold_hit = None
        self.success_given = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # 每個 episode 重設
        self.threshold_hit = {thr: False for thr, _ in self.threshold_rewards}
        self.success_given = False

        return obs, info

    def _extract_positions(self, obs):
        obs = np.asarray(obs)

        # 你需要依 MetaWorld observation 結構確認 target_pos index
        # 這裡先沿用你原本的假設寫法
        eef_pos = obs[:3]
        target_pos = obs[36:39]

        return eef_pos, target_pos

    def step(self, action):
        obs, dense_reward, terminated, truncated, info = self.env.step(action)

        success = bool(info.get("success", False))
        eef_pos, target_pos = self._extract_positions(obs)
        distance = np.linalg.norm(eef_pos - target_pos)

        reward = self.failure_reward
        reward_source = "failure"

        # success reward 每個 episode 只給一次
        if success and not self.success_given:
            reward = self.success_reward
            self.success_given = True
            reward_source = "success"

        else:
            matched = False

            # threshold reward 每層每個 episode 只給一次
            for threshold, threshold_reward in self.threshold_rewards:
                if distance < threshold and not self.threshold_hit[threshold]:
                    reward = threshold_reward
                    self.threshold_hit[threshold] = True
                    matched = True
                    reward_source = f"threshold_{threshold}"
                    break

            # 如果沒有命中任何新 threshold，再決定是否給 dense fallback
            if not matched and reward_source == "failure" and self.use_dense_fallback:
                reward = self.dense_coef * dense_reward
                reward_source = "dense_fallback"

        info["dense_reward"] = float(dense_reward)
        info["goal_distance"] = float(distance)
        info["threshold_reward"] = float(reward)
        info["reward_source"] = reward_source

        return obs, reward, terminated, truncated, info