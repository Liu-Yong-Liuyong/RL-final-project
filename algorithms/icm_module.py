import torch
import torch.nn as nn
import torch.nn.functional as F


class ICMModule(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        action_type="discrete",
        feature_dim=64,
        hidden_dim=128,
        device="cpu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_type = action_type
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

    def _prepare_action_input(self, action):
        """
        將 action 轉成 forward model 可接受的格式

        discrete:
            action: [B] -> one-hot [B, action_dim]

        continuous:
            action: [B, action_dim] -> 直接使用
        """
        if self.action_type == "discrete":
            if action.dim() == 2 and action.shape[-1] == self.action_dim:
                return action.float()
            return F.one_hot(action.long(), num_classes=self.action_dim).float()

        elif self.action_type == "continuous":
            return action.float()

        else:
            raise ValueError(f"Unsupported action_type: {self.action_type}")

    def forward_dynamics(self, phi_s, action_input):
        x = torch.cat([phi_s, action_input], dim=-1)
        return self.forward_model(x)

    def inverse_dynamics(self, phi_s, phi_next):
        x = torch.cat([phi_s, phi_next], dim=-1)
        return self.inverse_model(x)

    @torch.no_grad()
    def compute_intrinsic_reward(self, obs, action, next_obs):
        """
        obs: np.ndarray or tensor, shape [obs_dim]
        action:
            discrete -> int
            continuous -> np.ndarray / tensor, shape [action_dim]
        next_obs: np.ndarray or tensor, shape [obs_dim]
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        if self.action_type == "discrete":
            action_t = torch.as_tensor([action], dtype=torch.long, device=self.device)
        else:
            action_t = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)

        phi_s = self.encode(obs_t)
        phi_next = self.encode(next_obs_t)

        action_input = self._prepare_action_input(action_t)
        phi_next_pred = self.forward_dynamics(phi_s, action_input)

        intrinsic_reward = 0.5 * ((phi_next_pred - phi_next) ** 2).sum(dim=-1)
        return float(intrinsic_reward.item())

    def compute_loss(self, obs_batch, action_batch, next_obs_batch, beta=0.2):
        """
        obs_batch: tensor [B, obs_dim]
        action_batch:
            discrete   -> [B]
            continuous -> [B, action_dim]
        next_obs_batch: tensor [B, obs_dim]
        """
        phi_s = self.encode(obs_batch)
        phi_next = self.encode(next_obs_batch)

        # inverse model
        pred_action = self.inverse_dynamics(phi_s, phi_next)

        if self.action_type == "discrete":
            inverse_loss = F.cross_entropy(pred_action, action_batch.long())
        elif self.action_type == "continuous":
            inverse_loss = F.mse_loss(pred_action, action_batch.float())
        else:
            raise ValueError(f"Unsupported action_type: {self.action_type}")

        # forward model
        action_input = self._prepare_action_input(action_batch)
        phi_next_pred = self.forward_dynamics(phi_s, action_input)
        forward_loss = F.mse_loss(phi_next_pred, phi_next.detach())

        icm_loss = (1 - beta) * inverse_loss + beta * forward_loss

        return {
            "icm_loss": icm_loss,
            "inverse_loss": inverse_loss.detach().item(),
            "forward_loss": forward_loss.detach().item(),
        }


