#sample ppo agent, that can be used in evaluate.py for loading the pretrained model
import torch

class PPOAgent:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    @classmethod
    def load(cls, model_path, model_class):
        model = model_class()
        model.load_state_dict(torch.load(model_path))
        return cls(model)

    def predict(self, obs):
        with torch.no_grad():
            action = self.model.act(obs)
        return action
