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

#for adding the intrinsic reward from ICM module to the original environment extrinsic reward
class CuriosityRewardWrapper(gym.Wrapper):
    def __init__(self, env, icm_module, reward_scale=0.01):
        super().__init__(env)
        self.icm_module = icm_module #the module that will calculate intrinsic reward
        self.reward_scale = reward_scale

        self.prev_obs = None #since ICM module meed the observation transition (pre_obs, action, new_obs)
        self.transition_buffer = [] #for storing the transition

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_obs = obs
        return obs, info

    def step(self, action):
        next_obs, extrinsic_reward, terminated, truncated, info = self.env.step(action)

        intrinsic_reward = self.icm_module.compute_intrinsic_reward(
            self.prev_obs, action, next_obs
        )

        total_reward = extrinsic_reward + self.reward_scale * intrinsic_reward

        self.transition_buffer.append({
            "obs": self.prev_obs,
            "action": action,
            "next_obs": next_obs,
        })

        info["extrinsic_reward"] = float(extrinsic_reward)
        info["intrinsic_reward"] = float(intrinsic_reward)
        info["total_reward"] = float(total_reward)

        self.prev_obs = next_obs

        return next_obs, total_reward, terminated, truncated, info
    #for callback to update the ICM module
    def pop_transitions(self):
        transitions = self.transition_buffer
        self.transition_buffer = []
        return transitions

class RNDRewardWrapper(gym.Wrapper):
    def __init__(self, env, rnd_module, reward_scale=0.001):
        super().__init__(env)
        self.rnd_module = rnd_module
        self.reward_scale = reward_scale
        self.obs_buffer = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        next_obs, extrinsic_reward, terminated, truncated, info = self.env.step(action)

        intrinsic_reward = self.rnd_module.compute_intrinsic_reward(next_obs)
        total_reward = extrinsic_reward + self.reward_scale * intrinsic_reward

        self.obs_buffer.append(next_obs)

        info["extrinsic_reward"] = float(extrinsic_reward)
        info["intrinsic_reward"] = float(intrinsic_reward)
        info["total_reward"] = float(total_reward)

        return next_obs, total_reward, terminated, truncated, info

    def pop_observations(self):
        observations = self.obs_buffer
        self.obs_buffer = []
        return observations