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