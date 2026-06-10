import gymnasium as gym
import metaworld
import numpy as np
import random
from gymnasium.utils import seeding

from envs.base_env import BaseBenchmarkEnv


class MetaWorldButtonEnv(BaseBenchmarkEnv):
    def __init__(self, task_name, render_mode=None, fixed_seed=None):
        super().__init__(task_name=task_name)

        self.env = gym.make("Meta-World/MT1", env_name=task_name, render_mode=render_mode)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.fixed_seed = fixed_seed

    def reset(self, seed=None, options=None):
        if self.fixed_seed is not None:
            # === 1. 隔離防護罩：備份當前演算法的亂數進度 ===
            np_state = np.random.get_state()
            py_state = random.getstate()
            
            # === 2. 全局亂數歸零 ===
            np.random.seed(self.fixed_seed)
            random.seed(self.fixed_seed)
            
            # === 3. 核彈級破解：無視 Wrapper，直接摧毀並重建底層的亂數引擎 ===
            # 這是解決 MetaWorld 吃掉 seed 參數的唯一方法
            inner_np_random, _ = seeding.np_random(self.fixed_seed)
            self.env.unwrapped.np_random = inner_np_random
            if hasattr(self.env, "np_random"):
                self.env.np_random = inner_np_random
                
            # === 4. 執行重置 (此時底層引擎已被我們強制換成 seed 42) ===
            obs, info = self.env.reset(seed=self.fixed_seed, options=options)
            
            # === 5. 撤除防護罩：把亂數進度還給 RND/ICM ===
            np.random.set_state(np_state)
            random.setstate(py_state)
        else:
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
        hand_pos = obs[:3]  # obs[:3] is the position of the end effector of the robotic arm
        bins = np.round(hand_pos, 1)
        return tuple(bins.tolist())

    def is_success(self, obs, info=None) -> bool:
        return bool(info.get("success", 0.0)) if info is not None else False

    def close(self):
        self.env.close()

    def get_target_pos(self):
        return np.asarray(self.env.unwrapped._target_pos)