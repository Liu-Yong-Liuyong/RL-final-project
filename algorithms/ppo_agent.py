#sample ppo agent, that can be used in evaluate.py for loading the pretrained model
from stable_baselines3 import PPO

class PPOAgent:
    def __init__(self, model):
        self.model = model

    @classmethod
    def load(cls, model_path):
        model = PPO.load(model_path)
        return cls(model)

    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action
