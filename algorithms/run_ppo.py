#sample ppo
import argparse
from pathlib import Path
import yaml

from stable_baselines3 import PPO
from scripts.make_env import make_env


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_ppo_from_config(config: dict):
    env_cfg = config["env"]
    train_cfg = config["train"]
    output_cfg = config["output"]

    env_name = env_cfg["name"]
    env_kwargs = env_cfg.get("kwargs", {})

    env = make_env(env_name, **env_kwargs)

    save_path = output_cfg["save_path"]
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=train_cfg.get("learning_rate", 3e-4),
        gamma=train_cfg.get("gamma", 0.99),
        ent_coef=train_cfg.get("ent_coef", 0.0),
        verbose=train_cfg.get("verbose", 1),
    )

    model.learn(total_timesteps=train_cfg.get("total_timesteps", 50000))
    model.save(save_path)

    print(f"Model saved to {save_path}.zip")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train_ppo_from_config(config)


if __name__ == "__main__":
    main()
