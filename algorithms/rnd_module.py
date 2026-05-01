import torch
import torch.nn as nn
import torch.nn.functional as F


class RNDModule(nn.Module):
    def __init__(self, obs_dim, feature_dim=64, hidden_dim=128, device="cpu"):
        super().__init__()
        self.device = device

        self.target = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        # target network is fixed
        for p in self.target.parameters():
            p.requires_grad = False

        self.to(device)

    @torch.no_grad()
    def compute_intrinsic_reward(self, obs):
        """
        obs: np.ndarray or tensor, shape [obs_dim]
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        target_feat = self.target(obs_t)
        pred_feat = self.predictor(obs_t)

        intrinsic_reward = 0.5 * ((pred_feat - target_feat) ** 2).sum(dim=-1)
        return float(intrinsic_reward.item())

    def compute_loss(self, obs_batch):
        """
        obs_batch: tensor [B, obs_dim]
        """
        target_feat = self.target(obs_batch)
        pred_feat = self.predictor(obs_batch)

        rnd_loss = F.mse_loss(pred_feat, target_feat.detach())

        return {
            "rnd_loss": rnd_loss,
        }