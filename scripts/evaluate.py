import argparse
import yaml

from scripts.make_env import make_env
from scripts.make_agent import make_agent
from evaluation.evaluator import evaluate_agent
from wrappers.sparse_reward_wrappers import SparseRewardWrapper
from wrappers.sparse_reward_wrappers import GoalThresholdRewardWrapper #trying
from wrappers.sparse_reward_wrappers import PickPlaceMilestoneRewardWrapper #trying


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

'''
def build_env_from_config(config: dict, for_eval: bool = True):
    env_cfg = config["env"]
    env = make_env(
        env_cfg["name"],
        **env_cfg.get("kwargs", {})
    )

    wrappers_cfg = config.get("wrappers", {})

    sparse_cfg = wrappers_cfg.get("sparse_reward", {})
    if sparse_cfg.get("enabled", False):
        env = SparseRewardWrapper(env, **sparse_cfg.get("kwargs", {}))
        #env = GoalThresholdRewardWrapper(env, **sparse_cfg.get("kwargs", {}))
    # evaluation 時通常不要再套 exploration wrapper
    return env
'''
def build_env_from_config(config: dict, for_eval: bool = True):
    env_cfg = config["env"]
    wrappers_cfg = config.get("wrappers", {})

    env = make_env(
        env_name=env_cfg["name"],
        **env_cfg.get("kwargs", {})
    )

    sparse_cfg = wrappers_cfg.get("sparse_reward", {})
    if sparse_cfg.get("enabled", False):
        sparse_type = sparse_cfg.get("type", "basic")
        sparse_kwargs = sparse_cfg.get("kwargs", {})

        if sparse_type == "basic":
            env = SparseRewardWrapper(env, **sparse_kwargs)
        elif sparse_type == "goal_threshold":
            env = GoalThresholdRewardWrapper(env, **sparse_kwargs)
        elif sparse_type == "pick_place_milestone":
            env = PickPlaceMilestoneRewardWrapper(env, **sparse_kwargs)
        else:
            raise ValueError(f"Unknown sparse reward wrapper type: {sparse_type}")

    return env

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

    env = build_env_from_config(config, for_eval=True)

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