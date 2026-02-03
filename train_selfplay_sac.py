import logging
import os
from tqdm import tqdm
os.environ["RAY_DEDUP_LOGS"] = "1"
os.environ["RAY_LOG_TO_STDERR"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

# Suppress Ray's noisy warnings
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("ray.rllib").setLevel(logging.ERROR)
logging.getLogger("ray.tune").setLevel(logging.WARNING)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from argparse import ArgumentParser
from dataclasses import asdict

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec, DefaultModelConfig

from MultiAgentVersusEnv import FoosballSelfPlayEnv, FoosballSelfPlayEnvConfig


def env_creator(env_ctx):
    # env_ctx is dict-like (EnvContext). Use it to build your config.
    cfg = FoosballSelfPlayEnvConfig(**dict(env_ctx))
    return FoosballSelfPlayEnv(config=cfg)


def main():
    parser = ArgumentParser()
    # ... your args exactly as you have them ...
    parser.add_argument("--render-mode", type=str, default="none", choices=["none", "human", "rgb_array"])
    parser.add_argument("--policy_hz", type=float, default=200.0)
    parser.add_argument("--sim_hz", type=int, default=1000)
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_substeps", type=int, default=8)
    parser.add_argument("--speed_min", type=float, default=0.5)
    parser.add_argument("--speed_max", type=float, default=15.0)
    parser.add_argument("--bounce_prob", type=float, default=0.25)
    parser.add_argument("--cam_noise_std", type=float, default=0.0)
    parser.add_argument("--slider_vel_cap_mps", type=float, default=15.0)
    parser.add_argument("--kicker_vel_cap_rads", type=float, default=170.0)
    parser.add_argument("--real_time_gui", action="store_true")
    parser.add_argument("--serve_side", type=str, default="home", choices=["home", "away", "random"])
    args = parser.parse_args()

    env_config_obj = FoosballSelfPlayEnvConfig(
        render_mode=args.render_mode,
        policy_hz=args.policy_hz,
        sim_hz=args.sim_hz,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
        num_substeps=args.num_substeps,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
        bounce_prob=args.bounce_prob,
        cam_noise_std=args.cam_noise_std,
        slider_vel_cap_mps=args.slider_vel_cap_mps,
        kicker_vel_cap_rads=args.kicker_vel_cap_rads,
        real_time_gui=args.real_time_gui,
        serve_side=args.serve_side,
    )

    # Convert dataclass -> dict for Ray/RLlib
    env_config = asdict(env_config_obj)

    ray.init(ignore_reinit_error=True, configure_logging=False)
    register_env("FoosballSelfPlay", env_creator)

    rl_config = (
        SACConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env="FoosballSelfPlay", env_config=env_config, disable_env_checking=True)
        .env_runners(num_env_runners=1, num_envs_per_env_runner=8)
        .training(
            gamma=0.9,
            actor_lr=1e-3,
            critic_lr=2e-3,
            train_batch_size_per_learner=1024,
            num_steps_sampled_before_learning_starts=10000,
            replay_buffer_config={
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": 100_000,
            },
        )
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, *a, **k: "shared_policy",
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "shared_policy": RLModuleSpec(),
                }
            ),
        )
    )

    algo = rl_config.build()
    for i in tqdm(range(100)):
        result = algo.train()
        if i % 10 == 0:
            print(
                f"Iter {i:4d} | "
                f"ep_return_mean: {result['env_runners']['episode_return_mean']:.2f} | "
                f"ep_len_mean: {result['env_runners']['episode_len_mean']:.1f} | "
                # f"episodes: {result['env_runners']['num_episodes_lifetime']}"
            )

    algo.save("./checkpoints")


if __name__ == "__main__":
    main()
