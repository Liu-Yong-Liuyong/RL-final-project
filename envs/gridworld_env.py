import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import random
from envs.base_env import BaseBenchmarkEnv

class GridWorldEnv(BaseBenchmarkEnv):
    def __init__(self, width=10, height=10, start_pos=(0, 0), goal_pos=(9, 9), num_obstacles=0, random_seed=42, max_steps=100):
        super().__init__(task_name="SparseGridWorld")
        self.width = width
        self.height = height
        
        self.start_config = start_pos
        self.goal_config = goal_pos
        self.num_obstacles = num_obstacles
        self.random_seed = random_seed
        
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)
        
        self.start_pos = [0, 0]
        self.goal_pos = (9, 9)
        self.obstacles = set()
        self.current_pos = [0, 0]

        self.max_steps = max_steps
        self.current_step = 0

        self.fig = None
        self.ax = None
        self.img_plot = None

        self._generate_fixed_map()

    def _get_random_pos(self, exclude_set):
        while True:
            pos = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if pos not in exclude_set:
                return pos

    def _generate_fixed_map(self):
        if self.random_seed is not None:
            random.seed(self.random_seed)

        if self.goal_config == "random":
            self.goal_pos = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
        else:
            self.goal_pos = tuple(self.goal_config)

        if self.start_config == "random":
            self.start_pos = list(self._get_random_pos({self.goal_pos}))
        else:
            self.start_pos = list(self.start_config)

        self.obstacles = set()
        exclude = {tuple(self.start_pos), self.goal_pos}
        while len(self.obstacles) < self.num_obstacles:
            new_obs = self._get_random_pos(exclude)
            self.obstacles.add(new_obs)
            exclude.add(new_obs)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_pos = list(self.start_pos)

        self.current_step = 0
        
        obs = np.array([
            self.current_pos[0] / (self.width - 1),
            self.current_pos[1] / (self.height - 1)
        ], dtype=np.float32)
        info = {"success": False}
        info = self.build_info(obs, info)
        return obs, info

    def step(self, action):
        self.current_step += 1

        x, y = self.current_pos
        old_pos = (x, y)
        
        if action == 0: y = max(0, y - 1)                # up
        elif action == 1: x = min(self.width - 1, x + 1) # right
        elif action == 2: y = min(self.height - 1, y + 1)# down
        elif action == 3: x = max(0, x - 1)              # left
        
        new_pos = (x, y)
        
        if new_pos in self.obstacles:
            new_pos = old_pos
            
        self.current_pos = list(new_pos)
        obs = np.array(self.current_pos, dtype=np.int32)
        
        terminated = tuple(self.current_pos) == self.goal_pos
        reward = 1.0 if terminated else 0.0
        truncated = self.current_step >= self.max_steps
        
        info = {"success": terminated}
        info = self.build_info(obs, info)
        
        return obs, reward, terminated, truncated, info

    def get_coverage_id(self, obs):
        return tuple(self.current_pos)

    def is_success(self, obs, info=None):
        if info is not None and "success" in info:
            return bool(info["success"])
        return tuple(obs.tolist()) == self.goal_pos
    
    def render(self):
        img = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        
        color_empty = [230, 230, 230]
        color_agent = [161, 149, 252]
        color_goal  = [0, 194, 147]  
        color_wall  = [50, 50, 50]   
        color_start = [238, 134, 77] 
        
        img[:, :] = color_empty
        
        for (ox, oy) in self.obstacles:
            img[oy, ox] = color_wall
            
        sx, sy = self.start_pos
        img[sy, sx] = color_start
            
        gx, gy = self.goal_pos
        img[gy, gx] = color_goal
        
        cx, cy = self.current_pos
        img[cy, cx] = color_agent

        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(5, 5))
            self.img_plot = self.ax.imshow(img)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            plt.title(f"GridWorld (Seed: {self.random_seed})")
            plt.show(block=False)
        else:
            self.img_plot.set_data(img)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None