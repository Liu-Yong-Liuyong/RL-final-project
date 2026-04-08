'''
from scripts.make_env import make_env
from scripts.make_agent import make_agent
from evaluation.evaluator import evaluate_agent

def main():
    #-------------change to your task------------
    env_name = "frozenlake"
    env_kwargs = {
        "map_name": "4x4",
        "is_slippery": True,
    }
    #--------------------------------------------
    agent_name = "heuristic" #see make_agent.py

    env = make_env(env_name, **env_kwargs)
    agent = make_agent(agent_name, env_name, env)

    results = evaluate_agent(env, agent, num_episodes=10000, max_steps=200)

    print("===== Evaluation Results =====")
    print(f"Environment   : {env_name}")
    print(f"Agent         : {agent_name}")
    print(f"Average Return: {results['avg_return']:.4f}")
    print(f"Success Rate  : {results['success_rate']:.4f}")
    print(f"Avg Coverage  : {results['avg_coverage_count']:.4f}")
    print(f"Episodes      : {results['num_episodes']}")

if __name__ == "__main__":
    main()

'''
#when use pretrained model, it may look like this?
from scripts.make_env import make_env
from scripts.make_agent import make_agent
from evaluation.evaluator import evaluate_agent

def main():
    env_name = "frozenlake"
    env_kwargs = {"map_name": "4x4", "is_slippery": True}

    agent_name = "ppo"
    method_kwargs = {
        "model_path": "results/frozenlake_ppo_model"
    }

    env = make_env(env_name, **env_kwargs)
    agent = make_agent(agent_name, env_name, env, **method_kwargs)

    results = evaluate_agent(env, agent, num_episodes=20, max_steps=200)

    print(results)

if __name__ == "__main__":
    main()
