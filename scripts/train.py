import argparse
import yaml

from scripts.make_env import make_env
from algorithms.run_ppo import run_ppo

# 之後你 wrapper 實作好再打開
# from wrappers.sparse_reward_wrapper import SparseRewardWrapper
# from wrappers.exploration_wrapper import ExplorationWrapper


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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

    env_cfg = config["env"]
    env = make_env(
        env_name=env_cfg["name"],
        **env_cfg.get("kwargs", {})
    )

    wrappers_cfg = config.get("wrappers", {})

    sparse_cfg = wrappers_cfg.get("sparse_reward", {})
    if sparse_cfg.get("enabled", False):
        raise NotImplementedError("SparseRewardWrapper not connected yet.")
        # env = SparseRewardWrapper(env, **sparse_cfg.get("kwargs", {}))

    exploration_cfg = wrappers_cfg.get("exploration", {})
    if exploration_cfg.get("enabled", False):
        raise NotImplementedError("ExplorationWrapper not connected yet.")
        # env = ExplorationWrapper(
        #     env,
        #     method=exploration_cfg.get("method", "count_bonus"),
        #     **exploration_cfg.get("kwargs", {})
        # )

    agent_cfg = config["agent"]
    train_cfg = config["train"]
    output_cfg = config["output"]

    if agent_cfg["name"].lower() != "ppo":
        raise ValueError(f"Unsupported agent: {agent_cfg['name']}")

    run_ppo(
        env=env,
        total_timesteps=train_cfg["total_timesteps"],
        save_path=output_cfg["save_path"],
        ppo_kwargs=agent_cfg.get("kwargs", {}),
    )


if __name__ == "__main__":
    main()
