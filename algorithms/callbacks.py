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
        verbose=1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.num_eval_episodes = num_eval_episodes
        self.max_steps = max_steps
        self.alpha = alpha
        self.success_threshold = success_threshold

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

        if self.ewma_success_rate >= self.success_threshold:
            if self.verbose > 0:
                print(
                    f"[Early Stop] EWMA success rate "
                    f"{self.ewma_success_rate:.3f} >= {self.success_threshold:.3f}"
                )
            return False

        return True
