
import argparse
import random
import yaml
import numpy as np
import torch
import gymnasium as gym
import random
import numpy as np

from scripts.make_env import make_env
from algorithms.run_ppo import run_ppo
from algorithms.callbacks import EWMASuccessCallback

from wrappers.sparse_reward_wrappers import SparseRewardWrapper
from wrappers.sparse_reward_wrappers import GoalThresholdRewardWrapper #trying
from wrappers.exploration_wrapper import IntrinsicRewardWrapper
from wrappers.exploration_wrapper import CuriosityRewardWrapper
from wrappers.exploration_wrapper import RNDRewardWrapper
from algorithms.icm_module import ICMModule
from algorithms.rnd_module import RNDModule
from algorithms.icm_callback import ICMUpdateCallback
from algorithms.rnd_callback import RNDUpdateCallback
from stable_baselines3.common.callbacks import CallbackList
from wandb.integration.sb3 import WandbCallback

def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_env_from_config(config: dict):
    env_cfg = config["env"]
    wrappers_cfg = config.get("wrappers", {})

    env = make_env(
        env_name=env_cfg["name"],
        **env_cfg.get("kwargs", {})
    )

    sparse_cfg = wrappers_cfg.get("sparse_reward", {})
    if sparse_cfg.get("enabled", False):
        sparse_type = sparse_cfg.get("type", "basic")
        sparse_kwargs = sparse_cfg.get("kwargs", {})

        if sparse_type == "basic":
            env = SparseRewardWrapper(env, **sparse_kwargs)
        elif sparse_type == "goal_threshold":
            env = GoalThresholdRewardWrapper(env, **sparse_kwargs)
        else:
            raise ValueError(f"Unknown sparse reward wrapper type: {sparse_type}")

    return env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    seed = config.get("seed", 42)
    set_global_seeds(seed)

    env_cfg = config["env"]
    wrappers_cfg = config.get("wrappers", {})

    env = build_env_from_config(config)
    env.reset(seed=seed)

    callback_list = []

    exploration_cfg = wrappers_cfg.get("exploration", {})
    if exploration_cfg.get("enabled", False):
        method = exploration_cfg.get("method", None)

        if method == "intrinsic_count":
            env = IntrinsicRewardWrapper(
                env,
                **exploration_cfg.get("kwargs", {})
            )

        elif method == "icm":
            obs_dim = env.observation_space.shape[0]

            if isinstance(env.action_space, gym.spaces.Discrete):
                action_type = "discrete"
                action_dim = env.action_space.n
            elif isinstance(env.action_space, gym.spaces.Box):
                action_type = "continuous"
                action_dim = env.action_space.shape[0]
            else:
                raise ValueError(f"Unsupported action space for ICM: {type(env.action_space)}")

            icm_module = ICMModule(
                obs_dim=obs_dim,
                action_dim=action_dim,
                action_type=action_type,
                feature_dim=exploration_cfg.get("kwargs", {}).get("feature_dim", 64),
                hidden_dim=exploration_cfg.get("kwargs", {}).get("hidden_dim", 128),
                device=exploration_cfg.get("kwargs", {}).get("device", "cpu"),
            )

            env = CuriosityRewardWrapper(
                env,
                icm_module=icm_module,
                reward_scale=exploration_cfg.get("kwargs", {}).get("reward_scale", 0.01),
                clip_intrinsic=exploration_cfg.get("kwargs", {}).get("clip_intrinsic", 5.0),
            )

            icm_optimizer = torch.optim.Adam(
                icm_module.parameters(),
                lr=exploration_cfg.get("kwargs", {}).get("icm_lr", 1e-3),
            )

            icm_callback = ICMUpdateCallback(
                env_wrapper=env,
                icm_module=icm_module,
                icm_optimizer=icm_optimizer,
                beta=exploration_cfg.get("kwargs", {}).get("beta", 0.2),
                update_freq=exploration_cfg.get("kwargs", {}).get("update_freq", 1000),
                verbose=1,
            )
            callback_list.append(icm_callback)

        elif method == "rnd":
            obs_dim = env.observation_space.shape[0]

            rnd_module = RNDModule(
                obs_dim=obs_dim,
                feature_dim=exploration_cfg.get("kwargs", {}).get("feature_dim", 64),
                hidden_dim=exploration_cfg.get("kwargs", {}).get("hidden_dim", 128),
                device=exploration_cfg.get("kwargs", {}).get("device", "cpu"),
            )

            env = RNDRewardWrapper(
                env,
                rnd_module=rnd_module,
                reward_scale=exploration_cfg.get("kwargs", {}).get("reward_scale", 0.001),
            )

            rnd_optimizer = torch.optim.Adam(
                rnd_module.predictor.parameters(),
                lr=exploration_cfg.get("kwargs", {}).get("rnd_lr", 1e-3),
            )

            rnd_callback = RNDUpdateCallback(
                env_wrapper=env,
                rnd_module=rnd_module,
                rnd_optimizer=rnd_optimizer,
                update_freq=exploration_cfg.get("kwargs", {}).get("update_freq", 1000),
                verbose=1,
            )
            callback_list.append(rnd_callback)

        else:
            raise ValueError(f"Unknown exploration wrapper method: {method}")

    agent_cfg = config["agent"]
    train_cfg = config["train"]
    output_cfg = config["output"]

    if agent_cfg["name"].lower() != "ppo":
        raise ValueError(f"Unsupported agent: {agent_cfg['name']}")
    
    #add wandb
    wandb_cfg = config.get("wandb", {})
    if wandb_cfg.get("enabled", False):
        try:
            import wandb
            run = wandb.init(
                project=wandb_cfg.get("project", "rl-exploration-benchmark"),
                name=wandb_cfg.get("name", None),
                tags=wandb_cfg.get("tags", []),
                config=config,
                sync_tensorboard=True,
            )
            wandb.define_metric("global_step") 
            wandb.define_metric("*", step_metric="global_step")
            agent_cfg["kwargs"]["tensorboard_log"] = run.dir
            callback_list.append(WandbCallback(verbose=2))
        except ImportError:
            print("[WARNING] wandb not installed — skipping wandb logging. Run: pip install wandb")
            
    eval_env = build_env_from_config(config)
    eval_env.reset(seed=seed)

    ewma_callback = EWMASuccessCallback(
        eval_env=eval_env,
        eval_freq=5000,
        num_eval_episodes=20,
        max_steps=train_cfg.get("ewma_max_steps", 200),
        alpha=0.3,
        success_threshold=0.8,
        verbose=1,
        best_model_save_path=output_cfg["best_save_path"]
    )
    callback_list.append(ewma_callback)

    if len(callback_list) == 0:
        callback = None
    elif len(callback_list) == 1:
        callback = callback_list[0]
    else:
        callback = CallbackList(callback_list)

    run_ppo(
        env=env,
        total_timesteps=train_cfg["total_timesteps"],
        save_path=output_cfg["save_path"],
        ppo_kwargs=agent_cfg.get("kwargs", {}),
        callback=callback,
        seed=seed,
    )


if __name__ == "__main__":
    main()