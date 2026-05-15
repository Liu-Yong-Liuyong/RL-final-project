#if we use stable_baseline ppo, then we may write different exploration method wrappers here?
import math
import gymnasium as gym


class IntrinsicRewardWrapper(gym.Wrapper):
    def __init__(self, env, bonus_coef=0.01, mode="count_based"):
        super().__init__(env)
        self.bonus_coef = bonus_coef
        self.mode = mode
        self.state_counts = {}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        state_id = info.get("coverage_id", None)
        intrinsic_bonus = 0.0

        if state_id is not None:
            old_count = self.state_counts.get(state_id, 0)
            new_count = old_count + 1
            self.state_counts[state_id] = new_count

            if self.mode == "count_based":
                intrinsic_bonus = 1.0 / math.sqrt(new_count)
            else:
                raise ValueError(f"Unknown intrinsic reward mode: {self.mode}")

        total_reward = reward + self.bonus_coef * intrinsic_bonus

        info["extrinsic_reward"] = reward
        info["intrinsic_bonus"] = intrinsic_bonus
        info["total_reward"] = total_reward

        return obs, total_reward, terminated, truncated, info

'''
import numpy as np
import gymnasium as gym


class CuriosityRewardWrapper(gym.Wrapper):
    def __init__(self, env, icm_module, reward_scale=0.01, clip_intrinsic=None):
        super().__init__(env)
        self.icm_module = icm_module
        self.reward_scale = reward_scale
        self.clip_intrinsic = clip_intrinsic

        self.prev_obs = None
        self.transition_buffer = []

        # running stats for intrinsic reward
        self.int_count = 1e-4
        self.int_mean = 0.0
        self.int_M2 = 0.0  # for variance

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_obs = np.array(obs, copy=True)
        return obs, info

    def _update_running_stats(self, x: float):
        self.int_count += 1.0
        delta = x - self.int_mean
        self.int_mean += delta / self.int_count
        delta2 = x - self.int_mean
        self.int_M2 += delta * delta2

    def _get_running_std(self):
        var = self.int_M2 / self.int_count
        return np.sqrt(max(var, 1e-8))

    def _normalize_intrinsic_reward(self, intrinsic_reward: float):
        self._update_running_stats(intrinsic_reward)
        std = self._get_running_std()
        normalized = intrinsic_reward / (std + 1e-8)

        if self.clip_intrinsic is not None:
            normalized = np.clip(normalized, -self.clip_intrinsic, self.clip_intrinsic)

        return float(normalized)

    def step(self, action):
        next_obs, extrinsic_reward, terminated, truncated, info = self.env.step(action)

        intrinsic_reward = self.icm_module.compute_intrinsic_reward(
            self.prev_obs, action, next_obs
        )

        normalized_intrinsic_reward = self._normalize_intrinsic_reward(intrinsic_reward)

        total_reward = extrinsic_reward + self.reward_scale * normalized_intrinsic_reward

        self.transition_buffer.append({
            "obs": np.array(self.prev_obs, copy=True),
            "action": np.array(action, copy=True) if hasattr(action, "__len__") else action,
            "next_obs": np.array(next_obs, copy=True),
        })

        info["extrinsic_reward"] = float(extrinsic_reward)
        info["intrinsic_reward_raw"] = float(intrinsic_reward)
        info["intrinsic_reward"] = float(normalized_intrinsic_reward)
        info["total_reward"] = float(total_reward)

        if terminated or truncated:
            self.prev_obs = None
        else:
            self.prev_obs = np.array(next_obs, copy=True)

        return next_obs, total_reward, terminated, truncated, info

    def pop_transitions(self):
        transitions = self.transition_buffer
        self.transition_buffer = []
        return transitions
'''
import numpy as np
import gymnasium as gym


class CuriosityRewardWrapper(gym.Wrapper):
    def __init__(self, env, icm_module, reward_scale=0.01, clip_intrinsic=None):
        super().__init__(env)
        self.icm_module = icm_module
        self.reward_scale = reward_scale
        self.clip_intrinsic = clip_intrinsic

        self.prev_obs = None
        self.transition_buffer = []

        # running stats for intrinsic reward
        self.int_count = 1e-4
        self.int_mean = 0.0
        self.int_M2 = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_obs = np.array(obs, copy=True)
        return obs, info

    def _update_running_stats(self, x: float):
        self.int_count += 1.0
        delta = x - self.int_mean
        self.int_mean += delta / self.int_count
        delta2 = x - self.int_mean
        self.int_M2 += delta * delta2

    def _get_running_std(self):
        var = self.int_M2 / self.int_count
        return np.sqrt(max(var, 1e-8))

    def _process_intrinsic_reward(self, intrinsic_reward: float):
        """
        continuous action -> normalize
        discrete action   -> keep raw intrinsic reward
        """
        if getattr(self.icm_module, "action_type", "discrete") == "continuous":
            self._update_running_stats(intrinsic_reward)
            std = self._get_running_std()
            processed = intrinsic_reward / (std + 1e-8)
        else:
            processed = intrinsic_reward

        if self.clip_intrinsic is not None:
            processed = np.clip(processed, -self.clip_intrinsic, self.clip_intrinsic)

        return float(processed)

    def step(self, action):
        next_obs, extrinsic_reward, terminated, truncated, info = self.env.step(action)

        intrinsic_reward = self.icm_module.compute_intrinsic_reward(
            self.prev_obs, action, next_obs
        )

        processed_intrinsic_reward = self._process_intrinsic_reward(intrinsic_reward)

        total_reward = extrinsic_reward + self.reward_scale * processed_intrinsic_reward

        self.transition_buffer.append({
            "obs": np.array(self.prev_obs, copy=True),
            "action": np.array(action, copy=True) if hasattr(action, "__len__") else action,
            "next_obs": np.array(next_obs, copy=True),
        })

        info["extrinsic_reward"] = float(extrinsic_reward)
        info["intrinsic_reward_raw"] = float(intrinsic_reward)
        info["intrinsic_reward"] = float(processed_intrinsic_reward)
        info["total_reward"] = float(total_reward)

        if terminated or truncated:
            self.prev_obs = None
        else:
            self.prev_obs = np.array(next_obs, copy=True)

        return next_obs, total_reward, terminated, truncated, info

    def pop_transitions(self):
        transitions = self.transition_buffer
        self.transition_buffer = []
        return transitions

#normalized rnd
class RNDRewardWrapper(gym.Wrapper):
    def __init__(self, env, rnd_module, reward_scale=0.001):
        super().__init__(env)
        self.rnd_module = rnd_module
        self.reward_scale = reward_scale
        self.obs_buffer = []

        self.int_reward_mean = 0.0
        self.int_reward_var = 1.0
        self.int_reward_count = 1e-4

    def _update_running_stats(self, x):
        self.int_reward_count += 1
        delta = x - self.int_reward_mean
        self.int_reward_mean += delta / self.int_reward_count
        delta2 = x - self.int_reward_mean
        self.int_reward_var += delta * delta2

    def _normalize_intrinsic_reward(self, x):
        self._update_running_stats(x)
        std = float((self.int_reward_var / self.int_reward_count) ** 0.5)
        return float(x) / (std + 1e-8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        next_obs, extrinsic_reward, terminated, truncated, info = self.env.step(action)

        intrinsic_reward = float(self.rnd_module.compute_intrinsic_reward(next_obs))
        normalized_intrinsic_reward = self._normalize_intrinsic_reward(intrinsic_reward)#do the normalization makes it better

        total_reward = float(extrinsic_reward) + self.reward_scale * float(normalized_intrinsic_reward)

        self.obs_buffer.append(next_obs)

        info["extrinsic_reward"] = float(extrinsic_reward)
        info["intrinsic_reward_raw"] = intrinsic_reward
        info["intrinsic_reward"] = float(normalized_intrinsic_reward)
        info["total_reward"] = float(total_reward)

        return next_obs, total_reward, terminated, truncated, info

    def pop_observations(self):
        observations = self.obs_buffer
        self.obs_buffer = []
        return observations