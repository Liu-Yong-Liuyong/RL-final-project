#sample ppo agent,just for evaluate, not for training, that can be used in evaluate.py for loading the pretrained model
from stable_baselines3 import PPO
import numpy as np

class PPOAgent:
    def __init__(self, model):
        self.model = model

    @classmethod
    def load(cls, model_path):
        model = PPO.load(model_path)
        return cls(model)
    
    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)

        # For discrete-action envs like FrozenLake
        if isinstance(action, np.ndarray):
            if action.shape == ():      # scalar array
                return int(action.item())
            if action.size == 1:        # shape (1,) etc.
                return int(action.reshape(-1)[0])

        return action
    '''
    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action
    '''