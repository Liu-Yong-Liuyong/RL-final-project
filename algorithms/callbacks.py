import matplotlib
import matplotlib.figure
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from evaluation.evaluator import evaluate_agent
from algorithms.ppo_agent import PPOAgent


class EWMASuccessCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        eval_freq=5000,
        num_eval_episodes=20,
        max_steps=200,
        alpha=0.3,
        success_threshold=0.8,
        log_csv_path=None,
        verbose=1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.num_eval_episodes = num_eval_episodes
        self.max_steps = max_steps
        self.alpha = alpha
        self.success_threshold = success_threshold
        self.log_csv_path = log_csv_path

        self.ewma_success_rate = None
        self.eval_history = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        eval_agent = PPOAgent(self.model)

        results = evaluate_agent(
            env=self.eval_env,
            agent=eval_agent,
            num_episodes=self.num_eval_episodes,
            max_steps=self.max_steps,
        )

        current_success_rate = results["success_rate"]
        self.eval_history.append(current_success_rate)

        if self.ewma_success_rate is None:
            self.ewma_success_rate = current_success_rate
        else:
            self.ewma_success_rate = (
                self.alpha * current_success_rate
                + (1 - self.alpha) * self.ewma_success_rate
            )

        if self.verbose > 0:
            print(
                f"[Eval] steps={self.num_timesteps}, "
                f"success_rate={current_success_rate:.3f}, "
                f"ewma_success_rate={self.ewma_success_rate:.3f}, "
                f"avg_return={results['avg_return']:.3f}, "
                f"avg_coverage={results['avg_coverage_count']:.1f}"
            )

        try:
            import wandb
            if wandb.run is not None:
                wandb.log(
                    {
                        "eval/success_rate": current_success_rate,
                        "eval/ewma_success_rate": self.ewma_success_rate,
                        "eval/avg_return": results["avg_return"],
                        "eval/avg_coverage_count": results["avg_coverage_count"],
                    },
                    step=self.num_timesteps,
                )
        except ImportError:
            pass

        if self.log_csv_path is not None:
            self._save_csv_row(current_success_rate, results)

        if self.ewma_success_rate >= self.success_threshold:
            if self.verbose > 0:
                print(
                    f"[Early Stop] EWMA success rate "
                    f"{self.ewma_success_rate:.3f} >= {self.success_threshold:.3f}"
                )
            self._flush_csv()
            return False

        return True

    def _save_csv_row(self, success_rate, results):
        import csv, os
        write_header = not os.path.exists(self.log_csv_path)
        with open(self.log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["timestep", "success_rate", "ewma_success_rate",
                                 "avg_return", "avg_coverage_count"])
            writer.writerow([
                self.num_timesteps,
                round(success_rate, 6),
                round(self.ewma_success_rate, 6),
                round(results["avg_return"], 6),
                round(results["avg_coverage_count"], 2),
            ])

    def _flush_csv(self):
        pass  # rows written incrementally, nothing to flush


class CartPoleHeatmapCallback(BaseCallback):
    """
    Logs CartPole state visitation heatmap and per-episode metrics to wandb.

    Tracks (cart_position_bin, pole_angle_bin) from info["coverage_id"] during
    training rollouts and uploads a 2D heatmap image every `heatmap_freq` steps.
    Also logs episode return, length, success, and intrinsic reward whenever an
    episode ends.
    """

    def __init__(self, heatmap_freq: int = 10000, save_npy_path: str = None, verbose: int = 0):
        super().__init__(verbose)
        self.heatmap_freq = heatmap_freq
        self.save_npy_path = save_npy_path
        self.visit_counts: dict = {}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            cid = info.get("coverage_id")
            if cid is not None:
                self.visit_counts[cid] = self.visit_counts.get(cid, 0) + 1

            # SB3 stuffs a special "episode" key into info at episode end
            ep_info = info.get("episode")
            if ep_info is not None:
                self._log_episode(info, ep_info)

        if self.n_calls % self.heatmap_freq == 0 and self.visit_counts:
            self._log_heatmap()

        return True

    def _log_episode(self, info: dict, ep_info: dict):
        try:
            import wandb
            if wandb.run is None:
                return
        except ImportError:
            return

        log = {
            "train/ep_return": float(ep_info.get("r", 0.0)),
            "train/ep_length": int(ep_info.get("l", 0)),
        }
        if "success" in info:
            log["train/ep_success"] = float(info["success"])
        if "intrinsic_reward" in info:
            log["train/intrinsic_reward"] = float(info["intrinsic_reward"])
        if "intrinsic_reward_raw" in info:
            log["train/intrinsic_reward_raw"] = float(info["intrinsic_reward_raw"])

        import wandb
        wandb.log(log, step=self.num_timesteps)

    def _log_heatmap(self):
        try:
            import wandb
            if wandb.run is None:
                return
        except ImportError:
            return

        pos_bins = sorted(set(k[0] for k in self.visit_counts))
        angle_bins = sorted(set(k[1] for k in self.visit_counts))

        pos_idx = {v: i for i, v in enumerate(pos_bins)}
        angle_idx = {v: i for i, v in enumerate(angle_bins)}

        grid = np.zeros((len(angle_bins), len(pos_bins)))
        for (pos, angle), count in self.visit_counts.items():
            grid[angle_idx[angle], pos_idx[pos]] = count

        # Use Figure (not pyplot) to avoid display backend issues
        # Square figure so the heatmap panel matches teammate's style
        fig = matplotlib.figure.Figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
        fig.colorbar(im, ax=ax, label="Visit Count")

        # x-axis: cart position
        step = max(1, len(pos_bins) // 12)
        x_ticks = list(range(0, len(pos_bins), step))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{pos_bins[i]:.1f}" for i in x_ticks], rotation=45, fontsize=7)

        # y-axis: pole angle
        ax.set_yticks(range(len(angle_bins)))
        ax.set_yticklabels([f"{v:.1f}" for v in angle_bins], fontsize=7)

        # mark the target x=1.5 column
        target_col = min(pos_idx, key=lambda p: abs(p - 1.5))
        ax.axvline(x=pos_idx[target_col], color="deepskyblue",
                   linestyle="--", linewidth=1.5, label="target x≈1.5")

        ax.set_xlabel("Cart Position (binned, 0.1 steps)")
        ax.set_ylabel("Pole Angle rad (binned, 0.1 steps)")
        ax.set_title(f"State Visitation — step {self.num_timesteps:,}")
        ax.legend(fontsize=7)
        fig.tight_layout()

        import wandb
        wandb.log(
            {"train/state_visitation_heatmap": wandb.Image(fig)},
            step=self.num_timesteps,
        )
        fig.clf()

    def _on_training_end(self):
        if self.save_npy_path is not None and self.visit_counts:
            np.save(self.save_npy_path, self.visit_counts)
            print(f"[Heatmap] Visit counts saved to {self.save_npy_path}")
