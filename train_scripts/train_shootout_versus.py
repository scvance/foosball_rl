"""
train_shootout_versus.py

Self-play SAC training for ShootoutVersusEnv.

One policy plays both sides. Observations and actions are mirrored in the env
so the policy always sees the game from the "home" perspective.

Key design (mirrors sb3_selfplay.py):
- SB3 sees 2*n_underlying_envs virtual environments
- Even indices (0, 2, 4, ...) = home sides
- Odd indices  (1, 3, 5, ...) = away sides
- A single SAC policy acts on all virtual envs

Run:
    python train_scripts/train_shootout_versus.py

Then:
    tensorboard --logdir .
"""

import os
import time
import argparse
from collections import deque
import multiprocessing as mp
from multiprocessing import Process, Pipe
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.utils import set_random_seed

from foosball_envs.ShootoutVersusEnv import ShootoutVersusEnv


# ── Worker ────────────────────────────────────────────────────────────────────

def _worker(remote, parent_remote, env_fn):
    """Worker process running a single ShootoutVersusEnv."""
    parent_remote.close()
    env = env_fn()

    while True:
        try:
            cmd, data = remote.recv()
        except EOFError:
            break

        if cmd == "step":
            home_action, away_action = data
            obs_dict, reward_dict, terminated, truncated, info = env.step(
                {"home": home_action, "away": away_action}
            )
            done = terminated or truncated

            info["reward_home"] = reward_dict["home"]
            info["reward_away"] = reward_dict["away"]

            new_obs_dict = None
            if done:
                info["terminal_observation_home"] = obs_dict["home"].copy()
                info["terminal_observation_away"] = obs_dict["away"].copy()
                new_obs_dict, _ = env.reset()

            remote.send((
                obs_dict["home"], obs_dict["away"],
                reward_dict["home"], reward_dict["away"],
                done, info,
                new_obs_dict["home"] if new_obs_dict else None,
                new_obs_dict["away"] if new_obs_dict else None,
            ))

        elif cmd == "reset":
            obs_dict, info = env.reset()
            remote.send((obs_dict["home"], obs_dict["away"], info))

        elif cmd == "close":
            remote.close()
            break

        elif cmd == "get_spaces":
            remote.send((env.observation_space, env.action_space))

        elif cmd == "get_attr":
            remote.send(getattr(env, data, None))

        else:
            raise NotImplementedError(f"Unknown command: {cmd}")


# ── SelfPlayVecEnv ────────────────────────────────────────────────────────────

class SelfPlayVecEnv(VecEnv):
    """
    Vectorized wrapper exposing 2*n_underlying_envs virtual environments to SB3.

    Even indices = home observations/actions.
    Odd  indices = away observations/actions.
    """

    def __init__(self, env_fns: List):
        self.n_underlying_envs = len(env_fns)
        self.waiting = False
        self.closed = False

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.n_underlying_envs)])
        self.processes = []

        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            p = Process(target=_worker, args=(work_remote, remote, env_fn), daemon=True)
            p.start()
            self.processes.append(p)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        obs_space_dict, act_space_dict = self.remotes[0].recv()

        obs_space = obs_space_dict["home"]   # identical for both sides (mirrored)
        act_space = act_space_dict["home"]

        super().__init__(2 * self.n_underlying_envs, obs_space, act_space)

    def reset(self) -> np.ndarray:
        for remote in self.remotes:
            remote.send(("reset", None))

        results = [remote.recv() for remote in self.remotes]
        obs = np.zeros((self.num_envs,) + self.observation_space.shape, dtype=np.float32)

        for i, (home_obs, away_obs, _) in enumerate(results):
            obs[2 * i]     = home_obs
            obs[2 * i + 1] = away_obs

        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self.waiting = True
        for i, remote in enumerate(self.remotes):
            remote.send(("step", (actions[2 * i], actions[2 * i + 1])))

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        self.waiting = False
        results = [remote.recv() for remote in self.remotes]

        obs     = np.zeros((self.num_envs,) + self.observation_space.shape, dtype=np.float32)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones   = np.zeros(self.num_envs, dtype=bool)
        infos: List[Dict] = []

        for i, (h_obs, a_obs, h_rew, a_rew, done, info, new_h_obs, new_a_obs) in enumerate(results):
            # Home side
            home_info = info.copy()
            home_info["is_home"] = True
            if done and new_h_obs is not None:
                home_info["terminal_observation"] = info["terminal_observation_home"]
                obs[2 * i] = new_h_obs
            else:
                obs[2 * i] = h_obs
            rewards[2 * i] = h_rew
            dones[2 * i]   = done
            infos.append(home_info)

            # Away side (minimal info to avoid double-counting in stats callback)
            away_info = {"is_home": False}
            if done and new_a_obs is not None:
                away_info["terminal_observation"] = info["terminal_observation_away"]
                obs[2 * i + 1] = new_a_obs
            else:
                obs[2 * i + 1] = a_obs
            rewards[2 * i + 1] = a_rew
            dones[2 * i + 1]   = done
            infos.append(away_info)

        return obs, rewards, dones, infos

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.processes:
            p.join()
        self.closed = True

    def env_is_wrapped(self, wrapper_class, indices=None) -> List[bool]:
        return [False] * self.num_envs

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [None] * self.num_envs

    def get_attr(self, attr_name, indices=None):
        self.remotes[0].send(("get_attr", attr_name))
        val = self.remotes[0].recv()
        return [val] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        pass

    def seed(self, seed=None):
        return [None] * self.num_envs


# ── Env factory ───────────────────────────────────────────────────────────────

class EnvFactory:
    """Picklable factory so subprocess workers can construct envs."""
    def __init__(self, seed: int, idx: int, env_kwargs: Dict):
        self.seed = seed
        self.idx = idx
        self.env_kwargs = env_kwargs

    def __call__(self) -> ShootoutVersusEnv:
        return ShootoutVersusEnv(seed=self.seed + self.idx, **self.env_kwargs)


def create_selfplay_vec_env(n_envs: int, seed: int, **env_kwargs) -> SelfPlayVecEnv:
    env_fns = [EnvFactory(seed, i, env_kwargs) for i in range(n_envs)]
    return SelfPlayVecEnv(env_fns)


# ── Stats callback ────────────────────────────────────────────────────────────

class ShootoutStatsCallback(BaseCallback):
    """
    Logs shootout self-play statistics to TensorBoard.

    Events: "home_goal", "away_goal", "out", "stalled"
    Only processes info from home (even) virtual-env indices to avoid
    double-counting.
    """

    def __init__(self, rolling_window: int = 100, verbose: int = 0):
        super().__init__(verbose=verbose)
        w = rolling_window

        self.home_goals  = deque(maxlen=w)
        self.away_goals  = deque(maxlen=w)
        self.outs        = deque(maxlen=w)
        self.stalls      = deque(maxlen=w)

        self.home_rewards = deque(maxlen=w * 100)
        self.away_rewards = deque(maxlen=w * 100)

        self.total_home_goals  = 0
        self.total_away_goals  = 0
        self.total_outs        = 0
        self.total_stalls      = 0
        self.total_episodes    = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if not info.get("is_home", False):
                continue

            if "reward_home" in info:
                self.home_rewards.append(info["reward_home"])
                self.away_rewards.append(info["reward_away"])

            event = info.get("event")
            if event is None:
                continue

            self.total_episodes += 1

            is_home_goal = int(event == "home_goal")
            is_away_goal = int(event == "away_goal")
            is_out       = int(event == "out")
            is_stall     = int(event == "stalled")

            self.home_goals.append(is_home_goal)
            self.away_goals.append(is_away_goal)
            self.outs.append(is_out)
            self.stalls.append(is_stall)

            self.total_home_goals += is_home_goal
            self.total_away_goals += is_away_goal
            self.total_outs       += is_out
            self.total_stalls     += is_stall

        if self.total_episodes > 0:
            ep = self.total_episodes
            self.logger.record("shootout/total_episodes",   ep)
            self.logger.record("shootout/home_goals_total", self.total_home_goals)
            self.logger.record("shootout/away_goals_total", self.total_away_goals)
            self.logger.record("shootout/outs_total",       self.total_outs)
            self.logger.record("shootout/stalls_total",     self.total_stalls)

            if len(self.home_goals) > 0:
                self.logger.record("shootout/home_goal_rate", float(np.mean(self.home_goals)))
                self.logger.record("shootout/away_goal_rate", float(np.mean(self.away_goals)))
                self.logger.record("shootout/goal_rate",
                                   float(np.mean(self.home_goals)) + float(np.mean(self.away_goals)))
                self.logger.record("shootout/out_rate",   float(np.mean(self.outs)))
                self.logger.record("shootout/stall_rate", float(np.mean(self.stalls)))

        if len(self.home_rewards) > 0:
            self.logger.record("shootout/mean_home_reward", float(np.mean(self.home_rewards)))
            self.logger.record("shootout/mean_away_reward", float(np.mean(self.away_rewards)))

        return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir",          type=str,   default=".")
    parser.add_argument("--run_name",        type=str,   default=None)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--total_timesteps", type=int,   default=2_000_000)
    parser.add_argument("--n_envs",          type=int,   default=max(1, min(12, mp.cpu_count())))

    parser.add_argument("--eval_freq",       type=int,   default=50_000)
    parser.add_argument("--n_eval_episodes", type=int,   default=10)

    # Env params
    parser.add_argument("--policy_hz",          type=float, default=30.0)
    parser.add_argument("--sim_hz",             type=int,   default=240)
    parser.add_argument("--max_episode_steps",  type=int,   default=200)
    parser.add_argument("--serve_mode",         type=str,   default="random",
                        choices=["random_fire", "corner", "random"])
    parser.add_argument("--handle_vel_cap_mps", type=float, default=10.0)
    parser.add_argument("--paddle_vel_cap_rads",type=float, default=20.0)

    # Physics
    parser.add_argument("--ball_restitution",   type=float, default=0.30)
    parser.add_argument("--wall_restitution",   type=float, default=0.85)
    parser.add_argument("--paddle_restitution", type=float, default=0.85)
    parser.add_argument("--num_substeps",       type=int,   default=1)

    # SAC
    parser.add_argument("--device",         type=str,   default="auto")
    parser.add_argument("--learning_rate",  type=float, default=3e-4)
    parser.add_argument("--buffer_size",    type=int,   default=500_000)
    parser.add_argument("--learning_starts",type=int,   default=5_000)
    parser.add_argument("--batch_size",     type=int,   default=256)
    parser.add_argument("--tau",            type=float, default=0.005)
    parser.add_argument("--gamma",          type=float, default=0.99)
    parser.add_argument("--ckpt",           type=str,   default=None,
                        help="Path to a checkpoint .zip to resume from.")

    args = parser.parse_args()

    set_random_seed(args.seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"sac_shootout_{timestamp}"
    out_dir  = os.path.join(args.logdir, run_name)

    tb_dir   = os.path.join(out_dir, "tb")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    best_dir = os.path.join(out_dir, "best")
    for d in (out_dir, tb_dir, ckpt_dir, best_dir):
        os.makedirs(d, exist_ok=True)

    env_kwargs = dict(
        render_mode="none",
        policy_hz=args.policy_hz,
        sim_hz=args.sim_hz,
        max_episode_steps=args.max_episode_steps,
        serve_mode=args.serve_mode,
        handle_vel_cap_mps=args.handle_vel_cap_mps,
        paddle_vel_cap_rads=args.paddle_vel_cap_rads,
        ball_restitution=args.ball_restitution,
        wall_restitution=args.wall_restitution,
        paddle_restitution=args.paddle_restitution,
        num_substeps=args.num_substeps,
        real_time_gui=False,
    )

    print(f"Creating {args.n_envs} underlying envs ({2 * args.n_envs} virtual envs for SB3)...")

    train_env = create_selfplay_vec_env(n_envs=args.n_envs, seed=args.seed, **env_kwargs)
    train_env = VecMonitor(train_env)

    eval_env = create_selfplay_vec_env(n_envs=1, seed=args.seed + 10_000, **env_kwargs)
    eval_env = VecMonitor(eval_env)

    print(f"Observation space : {train_env.observation_space}")
    print(f"Action space      : {train_env.action_space}")
    print(f"Virtual envs (SB3): {train_env.num_envs}")

    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"   # MPS is often slower for small networks
    else:
        device = args.device
    print(f"Device: {device}")

    if args.ckpt is None:
        model = SAC(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=1,
            gradient_steps=1,
            verbose=0,
            tensorboard_log=tb_dir,
            device=device,
        )
    else:
        print(f"Loading checkpoint: {args.ckpt}")
        model = SAC.load(
            args.ckpt,
            env=train_env,
            device=device,
            tensorboard_log=tb_dir,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=ckpt_dir,
        name_prefix="sac_shootout",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=os.path.join(out_dir, "eval"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    stats_cb = ShootoutStatsCallback(rolling_window=100)
    callbacks = CallbackList([stats_cb, checkpoint_cb, eval_cb])

    print(f"\nTraining for {args.total_timesteps:,} timesteps...")
    print(f"Logs: {out_dir}")
    print("View with: tensorboard --logdir .\n")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name=run_name,
        progress_bar=True,
    )

    model.save(os.path.join(out_dir, "final_model"))
    train_env.close()
    eval_env.close()

    print("\nTraining complete!")
    print(f"Final model: {os.path.join(out_dir, 'final_model')}")


if __name__ == "__main__":
    main()
