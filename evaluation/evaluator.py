"""
Minimal evaluator for benchmark environments.
"""

from evaluation.metrics import (
    compute_average_return,
    compute_success_rate,
    compute_average_coverage,
)


def run_one_episode(env, agent, max_steps=1000):
    """
    Run one episode and collect benchmark statistics.

    Args:
        env: Gymnasium-compatible environment
        agent: Any object with agent.predict(obs)
        max_steps: Safety cap for episode length

    Returns:
        dict with episode_return, success, visited_ids, coverage_count
    """
    obs, info = env.reset()

    episode_return = 0.0
    visited_ids = set()
    visited_ids.add(info["coverage_id"])

    success = bool(info.get("success", False))
    done = False
    steps = 0

    while not done and steps < max_steps:
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        episode_return += float(reward)
        visited_ids.add(info["coverage_id"])
        success = success or bool(info.get("success", False))

        done = terminated or truncated
        steps += 1

    return {
        "episode_return": episode_return,
        "success": success,
        "visited_ids": visited_ids,
        "coverage_count": len(visited_ids),
        "num_steps": steps,
    }


def evaluate_agent(env, agent, num_episodes=10, max_steps=1000):
    """
    Evaluate an agent on an environment.

    Args:
        env: Gymnasium-compatible environment
        agent: Policy-like object with predict(obs)
        num_episodes: Number of episodes to evaluate
        max_steps: Max steps per episode

    Returns:
        Dictionary of aggregated metrics
    """
    episode_returns = []
    success_flags = []
    coverage_values = []

    for _ in range(num_episodes):
        result = run_one_episode(env, agent, max_steps=max_steps)
        episode_returns.append(result["episode_return"])
        success_flags.append(result["success"])
        coverage_values.append(result["coverage_count"])

    metrics = {
        "avg_return": compute_average_return(episode_returns),
        "success_rate": compute_success_rate(success_flags),
        "avg_coverage_count": compute_average_coverage(coverage_values),
        "num_episodes": num_episodes,
    }

    return metrics
