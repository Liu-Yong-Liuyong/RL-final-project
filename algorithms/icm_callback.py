from stable_baselines3.common.callbacks import BaseCallback
import torch

#update the ICM module
class ICMUpdateCallback(BaseCallback):
    def __init__(self, env_wrapper, icm_module, icm_optimizer, beta=0.2, update_freq=1000, verbose=1):
        super().__init__(verbose)
        self.env_wrapper = env_wrapper
        self.icm_module = icm_module
        self.icm_optimizer = icm_optimizer
        self.beta = beta
        self.update_freq = update_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq != 0:
            return True

        transitions = self.env_wrapper.pop_transitions()
        if len(transitions) == 0:
            return True

        obs_batch = torch.as_tensor([t["obs"] for t in transitions], dtype=torch.float32)
        action_batch = torch.as_tensor([t["action"] for t in transitions], dtype=torch.long)
        next_obs_batch = torch.as_tensor([t["next_obs"] for t in transitions], dtype=torch.float32)

        loss_dict = self.icm_module.compute_loss(
            obs_batch=obs_batch,
            action_batch=action_batch,
            next_obs_batch=next_obs_batch,
            beta=self.beta,
        )

        self.icm_optimizer.zero_grad()
        loss_dict["icm_loss"].backward()
        self.icm_optimizer.step()

        if self.verbose > 0:
            print(
                f"[ICM] steps={self.num_timesteps}, "
                f"icm_loss={loss_dict['icm_loss'].item():.4f}, "
                f"inverse_loss={loss_dict['inverse_loss']:.4f}, "
                f"forward_loss={loss_dict['forward_loss']:.4f}"
            )

        return True