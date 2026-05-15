"""
Factory function for creating benchmark environments.
"""
'''
from envs.gridworld_env import GridWorldEnv
from envs.frozenlake_env import FrozenLakeEnv
from envs.sparse_cartpole_env import SparseCartPoleEnv


def make_env(env_name: str, **kwargs):
    """
    Create an environment by name.

    Args:
        env_name: One of ["gridworld", "frozenlake", "cartpole"]
        **kwargs: Environment-specific arguments

    Returns:
        env: A Gymnasium-compatible environment
    """
    env_name = env_name.lower()

    if env_name == "gridworld":
        return GridWorldEnv(**kwargs)
    elif env_name == "frozenlake":
        return FrozenLakeEnv(**kwargs)
    elif env_name == "cartpole":
        return SparseCartPoleEnv(**kwargs)
    else:
        raise ValueError(f"Unknown environment: {env_name}")
'''

def make_env(env_name: str, **kwargs):
    env_name = env_name.lower()

    if env_name == "frozenlake":
        from envs.frozenlake_env import FrozenLakeEnv
        return FrozenLakeEnv(**kwargs)
    elif env_name == "gridworld":
        from envs.gridworld_env import GridWorldEnv
        return GridWorldEnv(**kwargs)
    elif env_name == "cartpole":
        from envs.sparse_cartpole_env import SparseCartPoleEnv
        return SparseCartPoleEnv(**kwargs)
    elif env_name == "metaworld":
        from envs.metaworld_reach_env import MetaWorldReachEnv
        return MetaWorldReachEnv(**kwargs)
    elif env_name == "metaworld_button":
        from envs.metaworld_button_env import MetaWorldButtonEnv
        return MetaWorldButtonEnv(**kwargs)
    else:
        raise ValueError(f"Unknown environment: {env_name}")
