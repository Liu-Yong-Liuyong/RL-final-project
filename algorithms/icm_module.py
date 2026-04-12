import torch
import torch.nn as nn
import torch.nn.functional as F


class ICMModule(nn.Module):
    def __init__(self, obs_dim, action_dim, feature_dim=64, hidden_dim=128, device="cpu"):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(),
        )

        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        self.to(device)

    def encode(self, obs):
        return self.encoder(obs)

    def forward_dynamics(self, phi_s, action_onehot):
        x = torch.cat([phi_s, action_onehot], dim=-1)
        return self.forward_model(x)

    def inverse_dynamics(self, phi_s, phi_next):
        x = torch.cat([phi_s, phi_next], dim=-1)
        return self.inverse_model(x)

    @torch.no_grad()
    def compute_intrinsic_reward(self, obs, action, next_obs):
        """
        obs: np.ndarray or tensor, shape [obs_dim]
        action: int
        next_obs: np.ndarray or tensor, shape [obs_dim]
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t = torch.as_tensor([action], dtype=torch.long, device=self.device)

        phi_s = self.encode(obs_t)
        phi_next = self.encode(next_obs_t)

        action_onehot = F.one_hot(action_t, num_classes=self.action_dim).float()
        phi_next_pred = self.forward_dynamics(phi_s, action_onehot)

        intrinsic_reward = 0.5 * ((phi_next_pred - phi_next) ** 2).sum(dim=-1)
        return float(intrinsic_reward.item())

    def compute_loss(self, obs_batch, action_batch, next_obs_batch, beta=0.2):
        """
        obs_batch: tensor [B, obs_dim]
        action_batch: tensor [B]
        next_obs_batch: tensor [B, obs_dim]
        """
        phi_s = self.encode(obs_batch)
        phi_next = self.encode(next_obs_batch)

        # inverse model
        pred_action_logits = self.inverse_dynamics(phi_s, phi_next)
        inverse_loss = F.cross_entropy(pred_action_logits, action_batch)

        # forward model
        action_onehot = F.one_hot(action_batch, num_classes=self.action_dim).float()
        phi_next_pred = self.forward_dynamics(phi_s, action_onehot)
        forward_loss = F.mse_loss(phi_next_pred, phi_next.detach())

        icm_loss = (1 - beta) * inverse_loss + beta * forward_loss

        return {
            "icm_loss": icm_loss,
            "inverse_loss": inverse_loss.detach().item(),
            "forward_loss": forward_loss.detach().item(),
        }