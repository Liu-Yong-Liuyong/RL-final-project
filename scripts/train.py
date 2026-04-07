'''
#sample structure of train.py
import os
import json
import yaml

from scripts.make_env import make_env
from scripts.make_agent import make_agent
from evaluation.evaluator import evaluate_agent


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_results(results, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main(config_path="configs/frozenlake_medium.yaml"):
    # 1. load config
    config = load_config(config_path)

    env_name = config["env_name"]
    env_kwargs = config.get("env_kwargs", {})

    method_name = config.get("method_name", "random")
    method_kwargs = config.get("method_kwargs", {})

    eval_cfg = config.get("eval", {})
    num_episodes = eval_cfg.get("num_episodes", 10)
    max_steps = eval_cfg.get("max_steps", 200)

    result_path = config.get("result_path", f"results/{env_name}_{method_name}.json")

    # 2. make environment
    env = make_env(env_name, **env_kwargs)

    # 3. make agent / method
    agent = make_agent(method_name, env_name, env, **method_kwargs)

    # 4. train if needed
    # for random / heuristic, no training is needed
    # for PPO, this part can later call run_ppo.py
    if hasattr(agent, "train"):
        print(f"[INFO] Training agent: {method_name}")
        agent.train(env, config)

    # 5. evaluate
    print(f"[INFO] Evaluating on {env_name} with method {method_name}")
    metrics = evaluate_agent(
        env=env,
        agent=agent,
        num_episodes=num_episodes,
        max_steps=max_steps,
    )

    # 6. save / print results
    print("===== Training/Evaluation Results =====")
    print(f"Environment   : {env_name}")
    print(f"Method        : {method_name}")
    print(f"Average Return: {metrics['avg_return']:.4f}")
    print(f"Success Rate  : {metrics['success_rate']:.4f}")
    print(f"Avg Coverage  : {metrics['avg_coverage_count']:.4f}")
    print(f"Episodes      : {metrics['num_episodes']}")

    save_results(metrics, result_path)
    print(f"[INFO] Results saved to {result_path}")


if __name__ == "__main__":
    main()
'''
