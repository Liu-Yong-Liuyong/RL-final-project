from algorithms.baselines import (
    RandomAgent,
    FrozenLakeHeuristicAgent,
    GridWorldHeuristicAgent,
    CartPoleHeuristicAgent,
)
from algorithms.ppo_agent import PPOAgent 

def make_agent(agent_name, env_name, env):
    agent_name = agent_name.lower()
    env_name = env_name.lower()

    if agent_name == "random":
        return RandomAgent(env.action_space)

    if env_name == "frozenlake":
        if agent_name == "heuristic":
            return FrozenLakeHeuristicAgent(env.action_space)

    if env_name == "gridworld":
        if agent_name == "heuristic":
            return GridWorldHeuristicAgent(env.action_space, goal_pos=env.goal_pos)

    if env_name == "cartpole":
        if agent_name == "heuristic":
            return CartPoleHeuristicAgent(env.action_space)
            
    if agent_name == "ppo":
        model_path = kwargs["model_path"]
        return PPOAgent.load(model_path)

    raise ValueError(f"Unsupported agent '{agent_name}' for env '{env_name}'")
