from pathlib import Path
from stable_baselines3 import PPO

'''
def run_ppo(env, total_timesteps: int, save_path: str, ppo_kwargs: dict):
    """
    Train PPO on the given environment.

    Args:
        env: Gymnasium-compatible environment
        total_timesteps: Number of training timesteps
        save_path: Path to save the trained model (without .zip)
        ppo_kwargs: Keyword arguments passed to stable_baselines3.PPO
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model = PPO(env=env, **ppo_kwargs)
    model.learn(total_timesteps=total_timesteps)
    model.save(str(save_path))

    print(f"[INFO] Model saved to {save_path}.zip")
    return model
'''
from pathlib import Path
from stable_baselines3 import PPO


def run_ppo(env, total_timesteps: int, save_path: str, ppo_kwargs: dict, callback=None):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model = PPO(env=env, **ppo_kwargs)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(str(save_path))

    print(f"[INFO] Model saved to {save_path}.zip")
    return model