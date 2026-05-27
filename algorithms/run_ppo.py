'''
from pathlib import Path
from stable_baselines3 import PPO

def run_ppo(env, total_timesteps: int, save_path: str, ppo_kwargs: dict, callback=None, seed=None):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model = PPO(env=env, seed=seed, **ppo_kwargs)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(str(save_path))

    print(f"[INFO] Model saved to {save_path}.zip")
    return model
'''
from pathlib import Path
from stable_baselines3 import PPO
import json
from wrappers.exploration_wrapper import ExplorationLoggingWrapper


def find_wrapper(env, wrapper_type):
    current = env
    while True:
        if isinstance(current, wrapper_type):
            return current
        if not hasattr(current, "env"):
            return None
        current = current.env

def run_ppo(env, total_timesteps: int, save_path: str, ppo_kwargs: dict, callback=None, seed=None):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model = PPO(env=env, seed=seed, **ppo_kwargs)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(str(save_path))

    log_wrapper = find_wrapper(env, ExplorationLoggingWrapper)
    if log_wrapper is not None:
        log_dir = Path("exploration_logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        #log_path = log_dir / "none_exploration_frozenlake_log.json"
        log_path = log_dir / "none_dc0_exploration_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_wrapper.exploration_log, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Exploration log saved to {log_path}")

    print(f"[INFO] Model saved to {save_path}.zip")
    return model