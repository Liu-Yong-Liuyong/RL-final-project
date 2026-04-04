"""
Base environment interface for RL exploration benchmark.

This file defines a lightweight base class to ensure:
1. Consistent Gymnasium API (reset / step)
2. Unified info dictionary format
3. Standard hooks for coverage and success

All environments should either inherit this class
or follow the same interface strictly.
"""

from abc import ABC, abstractmethod
import gymnasium as gym
from typing import Any, Dict, Hashable, Tuple


class BaseBenchmarkEnv(gym.Env, ABC):
    """
    Lightweight base class for all benchmark environments.
    """

    def __init__(self, task_name: str):
        super().__init__()
        self.task_name = task_name

    # =========================
    # Required Gym API
    # =========================
    @abstractmethod
    def reset(self, seed: int = None, options: dict = None) -> Tuple[Any, Dict]:
        """
        Reset environment state.

        Must return:
            obs, info
        """
        pass

    @abstractmethod
    def step(self, action) -> Tuple[Any, float, bool, bool, Dict]:
        """
        Take one step in the environment.

        Must return:
            obs, reward, terminated, truncated, info
        """
        pass

    # =========================
    # Benchmark-specific hooks
    # =========================
    @abstractmethod
    def get_coverage_id(self, obs) -> Hashable:
        """
        Return a hashable identifier for coverage tracking.

        Examples:
            GridWorld → (x, y)
            FrozenLake → int(state)
            CartPole → (angle_bin, angvel_bin)
        """
        pass

    @abstractmethod
    def is_success(self, obs, info=None) -> bool:
        """
        Define whether the current episode is successful.

        Examples:
            GridWorld → reached goal
            FrozenLake → reached G
            CartPole → survived enough steps
        """
        pass

    # =========================
    # Helper: build info dict
    # =========================
    def build_info(self, obs, info: Dict = None) -> Dict:
        """
        Standardize info dictionary across all environments.
        """
        if info is None:
            info = {}

        info["task_name"] = self.task_name
        info["coverage_id"] = self.get_coverage_id(obs)
        info["success"] = self.is_success(obs, info)

        return info
