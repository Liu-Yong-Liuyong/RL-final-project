"""
Metric utilities for benchmark evaluation.
"""


def compute_average_return(episode_returns):
    """
    Compute mean episodic return.
    """
    if len(episode_returns) == 0:
        return 0.0
    return sum(episode_returns) / len(episode_returns)


def compute_success_rate(success_flags):
    """
    Compute success rate from a list of booleans.
    """
    if len(success_flags) == 0:
        return 0.0
    return sum(bool(x) for x in success_flags) / len(success_flags)


def compute_average_coverage(coverage_values):
    """
    Compute mean coverage across episodes.
    """
    if len(coverage_values) == 0:
        return 0.0
    return sum(coverage_values) / len(coverage_values)
