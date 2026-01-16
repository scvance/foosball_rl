"""
Minimal self-play training loop for two SAC agents on FoosballVersusEnv.

This uses Stable-Baselines3 SAC models but drives rollouts manually so two
policies can act simultaneously in the same environment.
"""

import argparse
import os
from typing import Tuple

import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from gymnasium import spaces

try:
    from tqdm import tqdm
except ImportError:  # lightweight fallback if tqdm isn't installed
    tqdm = None

from FoosballVersusEnv import FoosballVersusEnv


class StaticSpaceEnv(gym.Env):
    """
    Minimal Gymnasium env to provide spaces for SB3 policy construction.
    We never really step it, but implement step/reset to satisfy VecEnv.
    """

    metadata = {"render_modes": []}

    def __init__(self, obs_space, act_space):
        super().__init__()
        self.observation_space = obs_space
        self.action_space = act_space

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0
        terminated = True
        truncated = True
        info = {}
        return obs, reward, terminated, truncated, info


def build_model(name: str, obs_space, act_space, device: str, lr: float, tensorboard: str) -> SAC:
    dummy_env = DummyVecEnv([lambda: StaticSpaceEnv(obs_space, act_space)])
    model = SAC(
        policy="MlpPolicy",
        env=dummy_env,
        learning_rate=lr,
        device=device,
        verbose=1,
        tensorboard_log=tensorboard,
        buffer_size=500_000,
        learning_starts=0,  # handled in custom loop
    )
    model._custom_name = name  # for logging
    return model


def add_transition(model: SAC, obs, action, reward, next_obs, done: bool, truncated: bool):
    info = {"TimeLimit.truncated": truncated}
    obs_b = np.expand_dims(obs, axis=0)
    next_obs_b = np.expand_dims(next_obs, axis=0)
    action_b = np.expand_dims(action, axis=0)
    reward_b = np.array([reward], dtype=np.float32)
    done_b = np.array([done], dtype=bool)
    model.replay_buffer.add(
        obs=obs_b,
        next_obs=next_obs_b,
        action=action_b,
        reward=reward_b,
        done=done_b,
        infos=[info],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_steps", type=int, default=200_000)
    parser.add_argument("--learning_starts", type=int, default=5_000)
    parser.add_argument("--train_freq", type=int, default=1_000)
    parser.add_argument("--gradient_steps", type=int, default=1_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="selfplay_models")
    parser.add_argument("--save_every", type=int, default=50_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    set_random_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # Env
    env = FoosballVersusEnv(render_mode="none", serve_side="random", seed=args.seed)
    obs_space = env.observation_space["home"]
    act_space = env.action_space["home"]

    # Models
    model_home = build_model("home", obs_space, act_space, args.device, args.lr, tensorboard=args.save_dir)
    model_away = build_model("away", obs_space, act_space, args.device, args.lr, tensorboard=args.save_dir)

    # Initialize loggers to avoid missing _logger when calling train()
    model_home._setup_learn(total_timesteps=args.total_steps)
    model_away._setup_learn(total_timesteps=args.total_steps)

    obs_dict, _ = env.reset()
    ep = 0
    ep_rewards_home = 0.0
    ep_rewards_away = 0.0
    ep_steps = 0
    bar = tqdm(total=args.total_steps, dynamic_ncols=True) if tqdm is not None else None

    for step in range(1, args.total_steps + 1):
        action_home, _ = model_home.predict(obs_dict["home"], deterministic=False)
        action_away, _ = model_away.predict(obs_dict["away"], deterministic=False)

        next_obs, rewards, terminated, truncated, info = env.step({"home": action_home, "away": action_away})
        done = bool(terminated or truncated)

        ep_rewards_home += float(rewards["home"])
        ep_rewards_away += float(rewards["away"])
        ep_steps += 1

        add_transition(model_home, obs_dict["home"], action_home, rewards["home"], next_obs["home"], done, bool(truncated))
        add_transition(model_away, obs_dict["away"], action_away, rewards["away"], next_obs["away"], done, bool(truncated))

        obs_dict = next_obs
        if done:
            ep += 1
            comps = info.get("reward_components")
            dense_home = comps["home"]["dense_block"] if comps else 0.0
            dense_away = comps["away"]["dense_block"] if comps else 0.0
            msg = (
                f"ep {ep:05d} steps={ep_steps:4d} event={info.get('event')} "
                f"R_home={ep_rewards_home:+.3f} R_away={ep_rewards_away:+.3f} "
                f"dense_home={dense_home:.3f} dense_away={dense_away:.3f}"
            )
            if bar is not None:
                bar.set_description(msg)
            else:
                print(msg)
            ep_rewards_home = 0.0
            ep_rewards_away = 0.0
            ep_steps = 0
            obs_dict, _ = env.reset()

        if step > args.learning_starts and (step % args.train_freq == 0):
            model_home.train(args.gradient_steps, args.batch_size)
            model_away.train(args.gradient_steps, args.batch_size)

        if step % args.save_every == 0 or step == args.total_steps:
            home_path = os.path.join(args.save_dir, f"home_step{step}.zip")
            away_path = os.path.join(args.save_dir, f"away_step{step}.zip")
            model_home.save(home_path)
            model_away.save(away_path)
            print(f"[save] step {step} saved {home_path} and {away_path}")

        if bar is not None:
            bar.update(1)

    if bar is not None:
        bar.close()
    env.close()


if __name__ == "__main__":
    main()
