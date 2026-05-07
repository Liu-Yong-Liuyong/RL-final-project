import time
import torch

from envs.sparse_cartpole_env import SparseCartPoleEnv
from algorithms.run_ppo import run_ppo
from algorithms.ppo_agent import PPOAgent
from algorithms.baselines import RandomAgent, CartPoleHeuristicAgent
from evaluation.evaluator import evaluate_agent
from wrappers.exploration_wrapper import IntrinsicRewardWrapper, CuriosityRewardWrapper, RNDRewardWrapper
from algorithms.icm_module import ICMModule
from algorithms.rnd_module import RNDModule
from algorithms.icm_callback import ICMUpdateCallback
from algorithms.rnd_callback import RNDUpdateCallback

SMOKE_STEPS = 3000
EVAL_EPISODES = 3
MAX_STEPS = 500


def make_cartpole():
    return SparseCartPoleEnv(target_x=1.5)


# env sanity check
env = make_cartpole()
obs, info = env.reset(seed=0)
print("obs shape:", obs.shape)
print("action space:", env.action_space)
print("info:", info)

obs, reward, terminated, truncated, info = env.step(0)
print("reward:", reward, "terminated:", terminated, "truncated:", truncated)
print("coverage_id:", info["coverage_id"], "success:", info["success"])
env.close()

# random agent
env = make_cartpole()
results = evaluate_agent(env, RandomAgent(env.action_space), num_episodes=EVAL_EPISODES, max_steps=MAX_STEPS)
print("random:", results)

# heuristic agent
env = make_cartpole()
results = evaluate_agent(env, CartPoleHeuristicAgent(env.action_space), num_episodes=EVAL_EPISODES, max_steps=MAX_STEPS)
print("heuristic:", results)

# PPO + entropy regularization (baseline)
env = make_cartpole()
t0 = time.time()
run_ppo(env, SMOKE_STEPS, "checkpoints/smoke_cartpole_entropy",
        {"policy": "MlpPolicy", "learning_rate": 0.0003, "gamma": 0.99, "ent_coef": 0.1, "verbose": 0})
print(f"entropy trained in {time.time()-t0:.1f}s")
env = make_cartpole()
results = evaluate_agent(env, PPOAgent.load("checkpoints/smoke_cartpole_entropy.zip"), num_episodes=EVAL_EPISODES, max_steps=MAX_STEPS)
print("entropy eval:", results)

# PPO + ICM
env = make_cartpole()
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
icm = ICMModule(obs_dim=obs_dim, action_dim=action_dim, action_type="discrete", feature_dim=64, hidden_dim=128, device="cpu")
env = CuriosityRewardWrapper(env, icm_module=icm, reward_scale=0.001, clip_intrinsic=5.0)
icm_opt = torch.optim.Adam(icm.parameters(), lr=3e-4)
icm_cb = ICMUpdateCallback(env_wrapper=env, icm_module=icm, icm_optimizer=icm_opt, beta=0.2, update_freq=500, verbose=0)
t0 = time.time()
run_ppo(env, SMOKE_STEPS, "checkpoints/smoke_cartpole_icm",
        {"policy": "MlpPolicy", "learning_rate": 0.0003, "gamma": 0.99, "ent_coef": 0.01, "verbose": 0},
        callback=icm_cb)
print(f"icm trained in {time.time()-t0:.1f}s")
env = make_cartpole()
results = evaluate_agent(env, PPOAgent.load("checkpoints/smoke_cartpole_icm.zip"), num_episodes=EVAL_EPISODES, max_steps=MAX_STEPS)
print("icm eval:", results)

# PPO + RND
env = make_cartpole()
rnd = RNDModule(obs_dim=obs_dim, feature_dim=64, hidden_dim=128, device="cpu")
env = RNDRewardWrapper(env, rnd_module=rnd, reward_scale=0.001)
rnd_opt = torch.optim.Adam(rnd.predictor.parameters(), lr=1e-3)
rnd_cb = RNDUpdateCallback(env_wrapper=env, rnd_module=rnd, rnd_optimizer=rnd_opt, update_freq=500, verbose=0)
t0 = time.time()
run_ppo(env, SMOKE_STEPS, "checkpoints/smoke_cartpole_rnd",
        {"policy": "MlpPolicy", "learning_rate": 0.0003, "gamma": 0.99, "ent_coef": 0.01, "verbose": 0},
        callback=rnd_cb)
print(f"rnd trained in {time.time()-t0:.1f}s")
env = make_cartpole()
results = evaluate_agent(env, PPOAgent.load("checkpoints/smoke_cartpole_rnd.zip"), num_episodes=EVAL_EPISODES, max_steps=MAX_STEPS)
print("rnd eval:", results)

print("\nall smoke tests passed (success_rate=0.0 at 3000 steps is expected — that is normal)")
