from stable_baselines3.common.callbacks import BaseCallback
import torch
import os
import json

class RNDUpdateCallback(BaseCallback):
    def __init__(self, env_wrapper, rnd_module, rnd_optimizer, update_freq=1000, verbose=1):
        super().__init__(verbose)
        self.env_wrapper = env_wrapper
        self.rnd_module = rnd_module
        self.rnd_optimizer = rnd_optimizer
        self.update_freq = update_freq
        # self.saved_exploration_logs = [] #for 2d drawing
        self.save_dir = "exploration_logs"
        # self.save_path = os.path.join(self.save_dir, "rnd_button_exploration_log.json")
        self.save_path = os.path.join(self.save_dir, "rnd_20x20_exploration_obstacles_log.json")
        self.is_first_write = True

        os.makedirs(self.save_dir, exist_ok=True)
        with open(self.save_path, "w", encoding="utf-8") as f:
            f.write("[\n")
        if self.verbose > 0:
            print(f"[RND] Initialized exploration log file at {self.save_path}")

    def _on_step(self) -> bool:
        observations = self.env_wrapper.pop_observations()
        if len(observations) == 0:
            return True
        ##############################################
        exploration_logs = self.env_wrapper.pop_exploration_logs()
        with open(self.save_path, "a", encoding="utf-8") as f:
            for item in exploration_logs:
                item["global_step"] = self.num_timesteps
                item["method"] = "rnd"
                
                if not self.is_first_write:
                    f.write(",\n")
                else:
                    self.is_first_write = False
                
                json_str = json.dumps(item, ensure_ascii=False, indent=2)
                indented_str = "\n".join("  " + line for line in json_str.splitlines())
                f.write(indented_str)
        ##############################################
        if self.n_calls % self.update_freq != 0:
            return True

        obs_batch = torch.as_tensor(observations, dtype=torch.float32, device=self.rnd_module.device)

        loss_dict = self.rnd_module.compute_loss(obs_batch=obs_batch)

        self.rnd_optimizer.zero_grad()
        loss_dict["rnd_loss"].backward()
        self.rnd_optimizer.step()

        if self.verbose > 0:
            print(
                f"[RND] steps={self.num_timesteps}, "
                f"rnd_loss={loss_dict['rnd_loss'].item():.4f}"
            )

        try:
            import wandb
            if wandb.run is not None:
                wandb.log(
                    {
                        "rnd/rnd_loss": loss_dict['rnd_loss'].item(),
                    },
                )
        except ImportError:
            pass

        return True
    ############################################################3
    def _on_training_end(self) -> None:
        # save_dir = "exploration_logs"
        # os.makedirs(save_dir, exist_ok=True)

        # #save_path = os.path.join(save_dir, "rnd_exploration_frozenlake_log.json")
        # save_path = os.path.join(save_dir, "rnd_dc0_exploration_log.json")
        # with open(save_path, "w", encoding="utf-8") as f:
        #     json.dump(self.saved_exploration_logs, f, ensure_ascii=False, indent=2)
        if os.path.exists(self.save_path):
            with open(self.save_path, "a", encoding="utf-8") as f:
                f.write("\n]")

        print(f"[RND] Exploration log saved to {self.save_path}")
    ##########################################################