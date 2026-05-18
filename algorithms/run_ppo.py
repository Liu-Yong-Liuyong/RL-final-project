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