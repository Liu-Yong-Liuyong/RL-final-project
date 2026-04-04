"""
Minimal script to test one benchmark environment with a random agent.
"""

from scripts.make_env import make_env
from evaluation.evaluator import RandomAgent, evaluate_agent


def main():
    # ===== Change these lines to test different environments =====
    env_name = "frozenlake"
    env_kwargs = {
        # Example kwargs for FrozenLakeEnv
        # Your env class should accept these if you use them
        "map_name": "4x4",
        "is_slippery": True,
    }

    # ===== Create environment =====
    env = make_env(env_name, **env_kwargs)

    # ===== Create random baseline =====
    agent = RandomAgent(env.action_space)

    # ===== Evaluate =====
    results = evaluate_agent(
        env=env,
        agent=agent,
        num_episodes=10,
        max_steps=200,
    )

    print("===== Evaluation Results =====")
    print(f"Environment   : {env_name}")
    print(f"Average Return: {results['avg_return']:.4f}")
    print(f"Success Rate  : {results['success_rate']:.4f}")
    print(f"Avg Coverage  : {results['avg_coverage_count']:.4f}")
    print(f"Episodes      : {results['num_episodes']}")


if __name__ == "__main__":
    main()
