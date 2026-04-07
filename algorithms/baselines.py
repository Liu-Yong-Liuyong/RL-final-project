#just for simple test

import random
import numpy as np


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, obs):
        return self.action_space.sample()


class FrozenLakeHeuristicAgent:
    """
    Very simple heuristic for FrozenLake:
    prefer DOWN or RIGHT.
    FrozenLake actions:
        0 = LEFT, 1 = DOWN, 2 = RIGHT, 3 = UP
    """
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, obs):
        return random.choice([1, 2])


class GridWorldHeuristicAgent:
    """
    Example heuristic:
    move right first, then move down.
    Assumes obs = (x, y), goal = (goal_x, goal_y)
    """
    def __init__(self, action_space, goal_pos):
        self.action_space = action_space
        self.goal_pos = goal_pos

    def predict(self, obs):
        x, y = obs
        goal_x, goal_y = self.goal_pos

        if x < goal_x:
            return 1  # 假設 1 = RIGHT
        elif y < goal_y:
            return 2  # 假設 2 = DOWN
        else:
            return 0  # fallback


class CartPoleHeuristicAgent:
    """
    Very simple CartPole heuristic:
    push in the direction of the pole angle.
    obs = [x, x_dot, theta, theta_dot]
    action:
        0 = left
        1 = right
    """
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, obs):
        theta = obs[2]
        return 1 if theta > 0 else 0
