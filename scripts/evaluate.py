import argparse
import yaml

from scripts.make_env import make_env
from scripts.make_agent import make_agent
from evaluation.evaluator import evaluate_agent


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
    env_name = env_cfg["name"]
    env_kwargs = env_cfg.get("kwargs", {})

    env = make_env(env_name, **env_kwargs)

    agent_cfg = config["agent"]
    agent_name = agent_cfg["name"].lower()

    eval_cfg = config.get("eval", {})
    agent_kwargs = {}

    if agent_name == "ppo":
        model_path = eval_cfg.get("model_path", None)
        if model_path is None:
            raise ValueError("eval.model_path must be provided for PPO evaluation.")
        agent_kwargs["model_path"] = model_path

    agent = make_agent(agent_name, env_name, env, **agent_kwargs)

    results = evaluate_agent(
        env,
        agent,
        num_episodes=eval_cfg.get("num_episodes", 20),
        max_steps=eval_cfg.get("max_steps", 200),
    )

    print(results)


if __name__ == "__main__":
    main()
