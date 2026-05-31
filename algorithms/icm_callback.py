from stable_baselines3.common.callbacks import BaseCallback
import torch
import os
import json

class ICMUpdateCallback(BaseCallback):
    def __init__(
        self,
        env_wrapper,
        icm_module,
        icm_optimizer,
        beta=0.2,
        update_freq=1000,
        verbose=1,
    ):
        super().__init__(verbose)
        self.env_wrapper = env_wrapper
        self.icm_module = icm_module
        self.icm_optimizer = icm_optimizer
        self.beta = beta
        self.update_freq = update_freq

        self.saved_exploration_logs = [] #for 2d drawing

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq != 0:
            return True

        transitions = self.env_wrapper.pop_transitions()
        
        if len(transitions) == 0:
            return True

        ##############################################
        exploration_logs = self.env_wrapper.pop_exploration_logs()
        for item in exploration_logs:
            item["global_step"] = self.num_timesteps
            item["method"] = "icm"
            self.saved_exploration_logs.append(item)
        ##############################################

        device = self.icm_module.device

        obs_batch = torch.as_tensor(
            [t["obs"] for t in transitions],
            dtype=torch.float32,
            device=device,
        )

        next_obs_batch = torch.as_tensor(
            [t["next_obs"] for t in transitions],
            dtype=torch.float32,
            device=device,
        )

        if self.icm_module.action_type == "discrete":
            action_batch = torch.as_tensor(
                [t["action"] for t in transitions],
                dtype=torch.long,
                device=device,
            )
        elif self.icm_module.action_type == "continuous":
            action_batch = torch.as_tensor(
                [t["action"] for t in transitions],
                dtype=torch.float32,
                device=device,
            )
        else:
            raise ValueError(f"Unsupported action_type: {self.icm_module.action_type}")

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

        try:
            import wandb
            if wandb.run is not None:
                wandb.log(
                    {
                        "icm/icm_loss": loss_dict['icm_loss'].item(),
                        "icm/inverse_loss": loss_dict['inverse_loss'],
                        "icm/forward_loss": loss_dict['forward_loss'],
                    },
                )
        except ImportError:
            pass

        return True

    def _on_training_end(self) -> None:
        save_dir = "exploration_logs"
        os.makedirs(save_dir, exist_ok=True)

        #save_path = os.path.join(save_dir, "icm_exploration_frozenlake_log.json")
        save_path = os.path.join(save_dir, "icm_dc0_exploration_log.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.saved_exploration_logs, f, ensure_ascii=False, indent=2)

        print(f"[ICM] Exploration log saved to {save_path}")
