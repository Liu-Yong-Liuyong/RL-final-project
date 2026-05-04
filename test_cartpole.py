"""
Quick smoke test for the CartPole pipeline.
Runs each of the 4 project-spec baselines for 3000 timesteps each
to verify the full stack works before committing to a full run.

Usage:
    python test_cartpole.py
"""
import time
import torch
import gymnasium as gym

from envs.sparse_cartpole_env import SparseCartPoleEnv
from algorithms.run_ppo import run_ppo
from algorithms.baselines import RandomAgent, CartPoleHeuristicAgent
from evaluation.evaluator import evaluate_agent
from wrappers.exploration_wrapper import IntrinsicRewardWrapper, CuriosityRewardWrapper, RNDRewardWrapper
from algorithms.icm_module import ICMModule
from algorithms.rnd_module import RNDModule
from algorithms.icm_callback import ICMUpdateCallback
from algorithms.rnd_callback import RNDUpdateCallback

SMOKE_TIMESTEPS = 3000
EVAL_EPISODES = 3
MAX_STEPS = 500


def make_cartpole():
    return SparseCartPoleEnv()


def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


# ----------------------------------------------------------------
# Step 1: Verify the environment itself
# ----------------------------------------------------------------
section("ENV SANITY CHECK")
env = make_cartpole()
obs, info = env.reset(seed=0)
print(f"obs shape : {obs.shape}  dtype: {obs.dtype}")
print(f"action space: {env.action_space}")
print(f"obs space   : {env.observation_space}")
print(f"info keys   : {list(info.keys())}")

obs, reward, terminated, truncated, info = env.step(0)
print(f"step() → reward={reward}, terminated={terminated}, truncated={truncated}")
print(f"coverage_id={info['coverage_id']}, success={info['success']}")
env.close()
print("PASSED")


# ----------------------------------------------------------------
# Step 2: Baselines (random + heuristic) — no training needed
# ----------------------------------------------------------------
section("BASELINE: Random Agent")
env = make_cartpole()
agent = RandomAgent(env.action_space)
t0 = time.time()
results = evaluate_agent(env, agent, num_episodes=EVAL_EPISODES, max_steps=MAX_STEPS)
print(f"  success_rate={results['success_rate']:.2f}  avg_return={results['avg_return']:.2f}"
      f"  coverage={results['avg_coverage_count']:.1f}  time={time.time()-t0:.1f}s")
print("PASSED")

section("BASELINE: Heuristic Agent")
env = make_cartpole()
agent = CartPoleHeuristicAgent(env.action_space)
t0 = time.time()
results = evaluate_agent(env, agent, num_episodes=EVAL_EPISODES, max_steps=MAX_STEPS)
print(f"  success_rate={results['success_rate']:.2f}  avg_return={results['avg_return']:.2f}"
      f"  coverage={results['avg_coverage_count']:.1f}  time={time.time()-t0:.1f}s")
print("PASSED")


# ----------------------------------------------------------------
# Step 3: [1] Vanilla PPO
# ----------------------------------------------------------------
section("[1] Vanilla PPO (no exploration, ent_coef=0.0)")
env = make_cartpole()
t0 = time.time()
run_ppo(
    env=env,
    total_timesteps=SMOKE_TIMESTEPS,
    save_path="checkpoints/smoke_cartpole_vanilla",
    ppo_kwargs={"policy": "MlpPolicy", "learning_rate": 0.0003,
                "gamma": 0.99, "ent_coef": 0.0, "verbose": 0},
)
print(f"  Trained in {time.time()-t0:.1f}s")

env = make_cartpole()
from algorithms.ppo_agent import PPOAgent
agent = PPOAgent.load("checkpoints/smoke_cartpole_vanilla.zip")
results = evaluate_agent(env, agent, num_episodes=EVAL_EPISODES, max_steps=MAX_STEPS)
print(f"  success_rate={results['success_rate']:.2f}  avg_return={results['avg_return']:.2f}")
print("PASSED")


# ----------------------------------------------------------------
# Step 4: [2] PPO + Entropy Regularization
# ----------------------------------------------------------------
section("[2] PPO + Entropy Regularization (ent_coef=0.1)")
env = make_cartpole()
t0 = time.time()
run_ppo(
    env=env,
    total_timesteps=SMOKE_TIMESTEPS,
    save_path="checkpoints/smoke_cartpole_entropy",
    ppo_kwargs={"policy": "MlpPolicy", "learning_rate": 0.0003,
                "gamma": 0.99, "ent_coef": 0.1, "verbose": 0},
)
print(f"  Trained in {time.time()-t0:.1f}s")

env = make_cartpole()
agent = PPOAgent.load("checkpoints/smoke_cartpole_entropy.zip")
results = evaluate_agent(env, agent, num_episodes=EVAL_EPISODES, max_steps=MAX_STEPS)
print(f"  success_rate={results['success_rate']:.2f}  avg_return={results['avg_return']:.2f}")
print("PASSED")


# ----------------------------------------------------------------
# Step 5: [3] PPO + ICM
# ----------------------------------------------------------------
section("[3] PPO + ICM")
env = make_cartpole()

obs_dim = env.observation_space.shape[0]   # 4
action_dim = env.action_space.n             # 2

icm = ICMModule(obs_dim=obs_dim, action_dim=action_dim, action_type="discrete",
                feature_dim=64, hidden_dim=128, device="cpu")
env = CuriosityRewardWrapper(env, icm_module=icm, reward_scale=0.001, clip_intrinsic=5.0)

icm_optimizer = torch.optim.Adam(icm.parameters(), lr=3e-4)
icm_callback = ICMUpdateCallback(env_wrapper=env, icm_module=icm,
                                  icm_optimizer=icm_optimizer, beta=0.2,
                                  update_freq=500, verbose=0)
t0 = time.time()
run_ppo(
    env=env,
    total_timesteps=SMOKE_TIMESTEPS,
    save_path="checkpoints/smoke_cartpole_icm",
    ppo_kwargs={"policy": "MlpPolicy", "learning_rate": 0.0003,
                "gamma": 0.99, "ent_coef": 0.01, "verbose": 0},
    callback=icm_callback,
)
print(f"  Trained in {time.time()-t0:.1f}s")

env = make_cartpole()
agent = PPOAgent.load("checkpoints/smoke_cartpole_icm.zip")
results = evaluate_agent(env, agent, num_episodes=EVAL_EPISODES, max_steps=MAX_STEPS)
print(f"  success_rate={results['success_rate']:.2f}  avg_return={results['avg_return']:.2f}")
print("PASSED")


# ----------------------------------------------------------------
# Step 6: [4] PPO + RND
# ----------------------------------------------------------------
section("[4] PPO + RND")
env = make_cartpole()

obs_dim = env.observation_space.shape[0]

rnd = RNDModule(obs_dim=obs_dim, feature_dim=64, hidden_dim=128, device="cpu")
env = RNDRewardWrapper(env, rnd_module=rnd, reward_scale=0.001)

rnd_optimizer = torch.optim.Adam(rnd.predictor.parameters(), lr=1e-3)
rnd_callback = RNDUpdateCallback(env_wrapper=env, rnd_module=rnd,
                                  rnd_optimizer=rnd_optimizer,
                                  update_freq=500, verbose=0)
t0 = time.time()
run_ppo(
    env=env,
    total_timesteps=SMOKE_TIMESTEPS,
    save_path="checkpoints/smoke_cartpole_rnd",
    ppo_kwargs={"policy": "MlpPolicy", "learning_rate": 0.0003,
                "gamma": 0.99, "ent_coef": 0.01, "verbose": 0},
    callback=rnd_callback,
)
print(f"  Trained in {time.time()-t0:.1f}s")

env = make_cartpole()
agent = PPOAgent.load("checkpoints/smoke_cartpole_rnd.zip")
results = evaluate_agent(env, agent, num_episodes=EVAL_EPISODES, max_steps=MAX_STEPS)
print(f"  success_rate={results['success_rate']:.2f}  avg_return={results['avg_return']:.2f}")
print("PASSED")


# ----------------------------------------------------------------
print(f"\n{'='*55}")
print("  ALL SMOKE TESTS PASSED")
print(f"{'='*55}\n")
print("Note: success_rate will likely be 0.0 at 3000 steps — that is")
print("expected. This test only verifies the pipeline runs without errors.")
