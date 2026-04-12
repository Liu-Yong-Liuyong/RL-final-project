import argparse
import yaml
import torch

from scripts.make_env import make_env
from algorithms.run_ppo import run_ppo
from algorithms.callbacks import EWMASuccessCallback

# 之後你 wrapper 實作好再打開
# from wrappers.sparse_reward_wrapper import SparseRewardWrapper
from wrappers.exploration_wrapper import IntrinsicRewardWrapper
from wrappers.exploration_wrapper import CuriosityRewardWrapper
from wrappers.exploration_wrapper import RNDRewardWrapper
from algorithms.icm_module import ICMModule
from algorithms.rnd_module import RNDModule
from algorithms.icm_callback import ICMUpdateCallback
from algorithms.rnd_callback import RNDUpdateCallback
from stable_baselines3.common.callbacks import CallbackList


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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

    env_cfg = config["env"]
    env = make_env(
        env_name=env_cfg["name"],
        **env_cfg.get("kwargs", {})
    )

    wrappers_cfg = config.get("wrappers", {})

    sparse_cfg = wrappers_cfg.get("sparse_reward", {})
    if sparse_cfg.get("enabled", False):
        raise NotImplementedError("SparseRewardWrapper not connected yet.")
        # env = SparseRewardWrapper(env, **sparse_cfg.get("kwargs", {}))

    exploration_cfg = wrappers_cfg.get("exploration", {})
    callback_list = []
    if exploration_cfg.get("enabled", False):
        #raise NotImplementedError("ExplorationWrapper not connected yet.")
        method = exploration_cfg.get("method", None)

        if method == "intrinsic_count":
            env = IntrinsicRewardWrapper(
                env,
                **exploration_cfg.get("kwargs", {})
            )
        elif method == "icm":
            obs_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n   # FrozenLake / discrete action 用這個

            icm_module = ICMModule(
                obs_dim=obs_dim,
                action_dim=action_dim,
                feature_dim=exploration_cfg.get("kwargs", {}).get("feature_dim", 64),
                hidden_dim=exploration_cfg.get("kwargs", {}).get("hidden_dim", 128),
                device=exploration_cfg.get("kwargs", {}).get("device", "cpu"),
            )

            env = CuriosityRewardWrapper(
                env,
                icm_module=icm_module,
                reward_scale=exploration_cfg.get("kwargs", {}).get("reward_scale", 0.01),
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
    #=================================
    eval_env = make_env(
        env_name=env_cfg["name"],
        **env_cfg.get("kwargs", {})
    )
    ewma_callback = EWMASuccessCallback(
        eval_env=eval_env,
        eval_freq=5000,
        num_eval_episodes=20,
        max_steps=200,
        alpha=0.3,
        success_threshold=0.8,
        verbose=1,
    )
    callback_list.append(ewma_callback)

    # ================= merge callbacks =================
    if len(callback_list) == 0:
        callback = None
    elif len(callback_list) == 1:
        callback = callback_list[0]
    else:
        callback = CallbackList(callback_list)
    # ===================================================
    run_ppo(
        env=env,
        total_timesteps=train_cfg["total_timesteps"],
        save_path=output_cfg["save_path"],
        ppo_kwargs=agent_cfg.get("kwargs", {}),
        callback=callback,
    )
    #===============================
    '''
    run_ppo(
        env=env,
        total_timesteps=train_cfg["total_timesteps"],
        save_path=output_cfg["save_path"],
        ppo_kwargs=agent_cfg.get("kwargs", {}),
    )
    '''

if __name__ == "__main__":
    main()
