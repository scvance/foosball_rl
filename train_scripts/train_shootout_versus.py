"""
train_shootout_versus.py  —  Opponent-pool self-play training for ShootoutVersusEnv.

Phase 0 (bootstrap): scripted rule-based opponent controls the away side while the
learner (home) trains against a competent fixed opponent.

Phase 1 (pool self-play): opponent drawn from a pool:
  - scripted_frac  workers → ScriptedPolicy
  - recent_frac    workers → latest checkpoint
  - remainder      workers → random historical checkpoint

SB3 sees only n_envs (home sides). Away side is managed internally by each worker.

Run:
    python train_scripts/train_shootout_versus.py

Then:
    tensorboard --logdir .
"""

import os
import time
import random
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
    BaseCallback, CallbackList, EvalCallback,
)
from stable_baselines3.common.utils import set_random_seed

from foosball_envs.ShootoutVersusEnv import ShootoutVersusEnv


# ── Scripted Policy ────────────────────────────────────────────────────────────

class ScriptedPolicy:
    """
    Rule-based opponent. Pure numpy, SB3-compatible predict() interface.

    Obs layout (21 dims, always in home frame):
      [0:3]  ball est pos  (x, y, z)   — x<0 = near home side
      [3:6]  ball est vel  (vx, vy, vz)
      [6:9]  ball pred pos
      [9]    own paddle angle (wrapped)
      [10]   own handle pos  (metres, ±0.11)
      [11]   own paddle vel
      [12]   own handle vel
      [13:17] opp joints (same structure)
      [17]   intercept_y
      [18]   intercept_z
      [19]   intercept_x_plane
      [20]   intercept_time

    The paddle has a GAP at its centre; two bars sit at ±0.103 m from the
    handle/paddle link origin. The handle must be OFFSET so one bar aligns with
    the ball, not the gap.

    The active contact side (bar_side ∈ {+1, -1}) is chosen online each step
    based on which side can align to target_y best with the current handle range.
    Spin direction is then chosen from that side:
      forward spin sign = -bar_side
    so the tangential velocity at the selected bar points toward the opponent goal.
    """

    BAR_OFFSET = 0.103   # m from handle/paddle link origin to each bar
    HANDLE_MAX  = 0.11   # m

    def __init__(self, n_envs: int = 1):
        self.n_envs = n_envs

    def on_episode_start(self, env_idx: int = 0):
        """No per-episode state needed; kept for interface compatibility."""
        return None

    def predict(self, obs: np.ndarray, state=None, episode_start=None, deterministic=False):
        """
        obs: (n_envs, 21) or (21,)
        returns: (actions, None)  — actions shape (n_envs, 3) or (3,)
        """
        single = obs.ndim == 1
        if single:
            obs = obs[np.newaxis, :]

        n = obs.shape[0]
        actions = np.zeros((n, 3), dtype=np.float32)

        for i in range(n):
            o = obs[i]
            ball_y      = float(o[1])
            handle_pos  = float(o[10])   # current handle position (m)
            intercept_y = float(o[17])
            intercept_t = float(o[20])

            # Ball is heading toward our goal when intercept_t is valid
            ball_coming = intercept_t > 1e-3

            # Target y: use intercept prediction when ball is heading toward us
            target_y = intercept_y if ball_coming else ball_y

            # Choose which bar side (+/- offset) to connect with based on
            # achievable alignment under handle limits.
            cand = []
            for bar_side in (1.0, -1.0):
                ht = float(np.clip(
                    target_y - bar_side * self.BAR_OFFSET,
                    -self.HANDLE_MAX, self.HANDLE_MAX,
                ))
                bar_y = ht + bar_side * self.BAR_OFFSET
                align_err = abs(bar_y - target_y)
                travel = abs(handle_pos - ht)
                cand.append((align_err, travel, bar_side, ht))

            _, _, bar_side, handle_target = min(cand, key=lambda x: (x[0], x[1]))

            handle_pos_action = handle_target / self.HANDLE_MAX   # [-1, 1]
            handle_vel_action = 1.0                                # max speed

            aligned = abs(handle_pos - handle_target) < 0.045

            # Spin sign depends on which bar side is used for contact:
            # bar_side=+1 -> spin negative, bar_side=-1 -> spin positive.
            paddle_vel_action = (-bar_side) if (aligned and ball_coming) else 0.0

            actions[i] = [handle_pos_action, handle_vel_action, paddle_vel_action]

        if single:
            return actions[0], None
        return actions, None


# ── Worker ────────────────────────────────────────────────────────────────────

def _worker(remote, parent_remote, env_fn):
    """
    Worker subprocess running a single ShootoutVersusEnv.

    Manages its own opponent (ScriptedPolicy or loaded SAC model).
    SB3 only receives home obs/rewards; the away action is generated internally.

    Protocol
    --------
    ("reset", None)            → (home_obs, info)
    ("step", home_action)      → (home_obs, home_rew, done, info)
                                 When done=True:
                                   info["terminal_observation"] = terminal home obs
                                   home_obs                     = new-episode first obs
    ("set_opponent", tag/path) → no ack (fire-and-forget)
    ("get_spaces", None)       → (obs_space["home"], act_space["home"])
    ("get_attr", name)         → attribute value
    ("close", None)            → exit
    """
    parent_remote.close()
    env = env_fn()

    opponent: object = ScriptedPolicy(n_envs=1)
    opponent.on_episode_start(0)
    away_obs_cache: Optional[np.ndarray] = None

    def load_opponent(tag_or_path: str):
        if tag_or_path == "scripted":
            opp = ScriptedPolicy(n_envs=1)
            opp.on_episode_start(0)
            return opp
        # Load SAC on CPU — safest for subprocess inference on macOS
        return SAC.load(tag_or_path, device="cpu")

    while True:
        try:
            cmd, data = remote.recv()
        except EOFError:
            break

        if cmd == "step":
            home_action = data

            # Generate away action from cached observation
            if away_obs_cache is not None:
                away_action, _ = opponent.predict(away_obs_cache, deterministic=True)
            else:
                away_action = np.zeros(3, dtype=np.float32)

            obs_dict, reward_dict, terminated, truncated, info = env.step(
                {"home": home_action, "away": away_action}
            )
            done = terminated or truncated

            info["reward_home"] = reward_dict["home"]
            info["is_home"]     = True

            if done:
                # Store terminal obs before auto-reset
                info["terminal_observation"] = obs_dict["home"].copy()

                # Auto-reset
                new_obs_dict, _ = env.reset()
                away_obs_cache = new_obs_dict["away"].copy()
                home_obs       = new_obs_dict["home"].copy()

                if hasattr(opponent, "on_episode_start"):
                    opponent.on_episode_start(0)
            else:
                away_obs_cache = obs_dict["away"].copy()
                home_obs       = obs_dict["home"].copy()

            remote.send((home_obs, reward_dict["home"], done, info))

        elif cmd == "reset":
            obs_dict, info = env.reset()
            away_obs_cache = obs_dict["away"].copy()
            home_obs       = obs_dict["home"].copy()

            if hasattr(opponent, "on_episode_start"):
                opponent.on_episode_start(0)

            remote.send((home_obs, info))

        elif cmd == "set_opponent":
            # Fire-and-forget — no ack needed
            opponent = load_opponent(data)

        elif cmd == "get_spaces":
            remote.send((env.observation_space["home"], env.action_space["home"]))

        elif cmd == "get_attr":
            remote.send(getattr(env, data, None))

        elif cmd == "close":
            env.close()
            remote.close()
            break

        else:
            raise NotImplementedError(f"Unknown command: {cmd}")


# ── SelfPlayVecEnv ────────────────────────────────────────────────────────────

class SelfPlayVecEnv(VecEnv):
    """
    Vectorized wrapper exposing n_envs home sides to SB3.
    Away side is handled internally by each worker's opponent.
    """

    def __init__(self, env_fns: List):
        self.n_underlying_envs = len(env_fns)
        self.waiting = False
        self.closed  = False

        self.remotes, self.work_remotes = zip(
            *[Pipe() for _ in range(self.n_underlying_envs)]
        )
        self.processes = []

        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            p = Process(target=_worker, args=(work_remote, remote, env_fn), daemon=True)
            p.start()
            self.processes.append(p)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        obs_space, act_space = self.remotes[0].recv()

        super().__init__(self.n_underlying_envs, obs_space, act_space)

    def reset(self) -> np.ndarray:
        for remote in self.remotes:
            remote.send(("reset", None))

        results = [remote.recv() for remote in self.remotes]
        obs = np.zeros((self.num_envs,) + self.observation_space.shape, dtype=np.float32)
        for i, (home_obs, _) in enumerate(results):
            obs[i] = home_obs
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self.waiting = True
        for i, remote in enumerate(self.remotes):
            remote.send(("step", actions[i]))

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        self.waiting = False
        results = [remote.recv() for remote in self.remotes]

        obs     = np.zeros((self.num_envs,) + self.observation_space.shape, dtype=np.float32)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones   = np.zeros(self.num_envs, dtype=bool)
        infos: List[Dict] = []

        for i, (home_obs, home_rew, done, info) in enumerate(results):
            obs[i]     = home_obs   # new-episode obs when done=True
            rewards[i] = home_rew
            dones[i]   = done
            # terminal_observation is already in info when done=True (set by worker)
            infos.append(info)

        return obs, rewards, dones, infos

    def set_opponent(self, env_idx: int, path_or_tag: str) -> None:
        """
        Fire-and-forget: send set_opponent to a specific worker.
        Safe to call between step_wait() and step_async() when workers are idle.
        """
        self.remotes[env_idx].send(("set_opponent", path_or_tag))

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
        self.seed      = seed
        self.idx       = idx
        self.env_kwargs = env_kwargs

    def __call__(self) -> ShootoutVersusEnv:
        return ShootoutVersusEnv(seed=self.seed + self.idx, **self.env_kwargs)


def create_selfplay_vec_env(n_envs: int, seed: int, **env_kwargs) -> SelfPlayVecEnv:
    env_fns = [EnvFactory(seed, i, env_kwargs) for i in range(n_envs)]
    return SelfPlayVecEnv(env_fns)


# ── Stats Callback ────────────────────────────────────────────────────────────

class ShootoutStatsCallback(BaseCallback):
    """
    Logs shootout training statistics to TensorBoard.

    Events: "home_goal", "away_goal", "out", "stalled"
    Only home-side infos are present in the new design (is_home always True).
    """

    def __init__(self, rolling_window: int = 100, verbose: int = 0):
        super().__init__(verbose=verbose)
        w = rolling_window

        self.home_goals  = deque(maxlen=w)
        self.away_goals  = deque(maxlen=w)
        self.outs        = deque(maxlen=w)
        self.stalls      = deque(maxlen=w)

        self.home_rewards = deque(maxlen=w * 100)

        self.total_home_goals = 0
        self.total_away_goals = 0
        self.total_outs       = 0
        self.total_stalls     = 0
        self.total_episodes   = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if not info.get("is_home", False):
                continue

            if "reward_home" in info:
                self.home_rewards.append(info["reward_home"])

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

        return True


# ── Pool Checkpoint Callback ───────────────────────────────────────────────────

class PoolCheckpointCallback(BaseCallback):
    """
    Saves checkpoints on a fixed schedule and manages opponent-pool assignment.

    Phase 0 (timesteps < bootstrap_steps):
        All workers → scripted opponent.

    Phase 1 (timesteps >= bootstrap_steps):
        Workers are assigned from the pool each opponent_update_freq steps:
          round(n * scripted_frac)  → ScriptedPolicy
          round(n * recent_frac)    → latest checkpoint
          remainder                 → random historical checkpoint
        If no checkpoints exist yet, remaining slots fall back to scripted.

    Pool trim: keeps the oldest checkpoint plus the newest max_pool_size entries.
    """

    def __init__(
        self,
        train_env: SelfPlayVecEnv,
        ckpt_dir: str,
        bootstrap_steps: int = 200_000,
        checkpoint_freq: int = 50_000,
        opponent_update_freq: int = 10_000,
        scripted_frac: float = 0.2,
        recent_frac: float = 0.5,
        max_pool_size: int = 5,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.train_env           = train_env
        self.ckpt_dir            = ckpt_dir
        self.bootstrap_steps     = bootstrap_steps
        self.checkpoint_freq     = checkpoint_freq
        self.opponent_update_freq = opponent_update_freq
        self.scripted_frac       = scripted_frac
        self.recent_frac         = recent_frac
        self.max_pool_size       = max_pool_size

        self.checkpoints: List[str] = []
        self._last_ckpt_count    = 0
        self._last_update_count  = 0

    def _on_step(self) -> bool:
        t = self.num_timesteps

        # Checkpoint saving
        ckpt_count = t // self.checkpoint_freq
        if ckpt_count > self._last_ckpt_count:
            self._last_ckpt_count = ckpt_count
            self._save_checkpoint(t)

        # Opponent update
        update_count = t // self.opponent_update_freq
        if update_count > self._last_update_count:
            self._last_update_count = update_count
            self._update_opponents(t)

        return True

    def _save_checkpoint(self, t: int) -> None:
        path = os.path.join(self.ckpt_dir, f"sac_shootout_{t}_steps")
        self.model.save(path)
        full_path = path + ".zip"
        self.checkpoints.append(full_path)

        # Trim: keep first (oldest historical) + newest max_pool_size
        if len(self.checkpoints) > self.max_pool_size + 1:
            keep = [self.checkpoints[0]] + self.checkpoints[-self.max_pool_size:]
            self.checkpoints = keep

        if self.verbose >= 1:
            print(
                f"[PoolCkpt] Saved: {full_path}  "
                f"(pool size: {len(self.checkpoints)})"
            )

    def _update_opponents(self, t: int) -> None:
        n = self.train_env.n_underlying_envs

        if t < self.bootstrap_steps:
            if self.verbose >= 1:
                print(f"[PoolCkpt] Phase 0 (t={t:,}): all {n} workers → scripted")
            for i in range(n):
                self.train_env.set_opponent(i, "scripted")
            return

        # Phase 1: pool assignments
        n_scripted = round(n * self.scripted_frac)
        n_recent   = round(n * self.recent_frac)
        n_hist     = n - n_scripted - n_recent

        assignments = ["scripted"] * n_scripted

        if self.checkpoints:
            recent_path = self.checkpoints[-1]
            assignments += [recent_path] * n_recent

            # Historical pool: everything except the most recent
            hist_pool = self.checkpoints[:-1] if len(self.checkpoints) > 1 else self.checkpoints
            for _ in range(n_hist):
                assignments.append(random.choice(hist_pool))
        else:
            # No checkpoints yet — fall back all remaining slots to scripted
            assignments += ["scripted"] * (n_recent + n_hist)

        random.shuffle(assignments)

        if self.verbose >= 1:
            s_count = assignments.count("scripted")
            c_count = n - s_count
            print(
                f"[PoolCkpt] Phase 1 (t={t:,}): "
                f"{s_count} scripted, {c_count} checkpoint workers"
            )

        for i, tag_or_path in enumerate(assignments):
            self.train_env.set_opponent(i, tag_or_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Opponent-pool self-play SAC training for ShootoutVersusEnv."
    )
    parser.add_argument("--logdir",          type=str,   default=".")
    parser.add_argument("--run_name",        type=str,   default=None)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--total_timesteps", type=int,   default=2_000_000)
    parser.add_argument("--n_envs",          type=int,   default=max(1, min(12, mp.cpu_count())))

    parser.add_argument("--eval_freq",        type=int,   default=50_000)
    parser.add_argument("--n_eval_episodes",  type=int,   default=10)

    # Env params
    parser.add_argument("--policy_hz",           type=float, default=200.0)
    parser.add_argument("--sim_hz",              type=int,   default=1000)
    parser.add_argument("--max_episode_steps",   type=int,   default=200)
    parser.add_argument("--serve_mode",          type=str,   default="random",
                        choices=["random_fire", "corner", "random"])
    parser.add_argument("--handle_vel_cap_mps",  type=float, default=10.0)
    parser.add_argument("--paddle_vel_cap_rads", type=float, default=20.0)

    # Physics
    parser.add_argument("--ball_restitution",    type=float, default=0.30)
    parser.add_argument("--wall_restitution",    type=float, default=0.85)
    parser.add_argument("--paddle_restitution",  type=float, default=0.85)
    parser.add_argument("--num_substeps",        type=int,   default=1)

    # SAC
    parser.add_argument("--device",          type=str,   default="auto")
    parser.add_argument("--learning_rate",   type=float, default=3e-4)
    parser.add_argument("--buffer_size",     type=int,   default=500_000)
    parser.add_argument("--learning_starts", type=int,   default=5_000)
    parser.add_argument("--batch_size",      type=int,   default=256)
    parser.add_argument("--tau",             type=float, default=0.005)
    parser.add_argument("--gamma",           type=float, default=0.99)
    parser.add_argument("--ckpt",            type=str,   default=None,
                        help="Path to a .zip checkpoint to resume from.")

    # Pool self-play
    parser.add_argument("--bootstrap_steps",      type=int,   default=200_000,
                        help="Steps before switching to pool self-play.")
    parser.add_argument("--opponent_update_freq",  type=int,   default=10_000,
                        help="How often (steps) to reassign opponents.")
    parser.add_argument("--scripted_frac",         type=float, default=0.2,
                        help="Fraction of workers using scripted opponent in phase 1.")
    parser.add_argument("--recent_frac",           type=float, default=0.5,
                        help="Fraction of workers using the latest checkpoint.")
    parser.add_argument("--historical_frac",       type=float, default=0.3,
                        help="(Informational) remainder goes to historical checkpoints.")
    parser.add_argument("--max_pool_size",         type=int,   default=5,
                        help="Max historical checkpoints to keep in the pool.")

    args = parser.parse_args()

    set_random_seed(args.seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name  = args.run_name or f"sac_shootout_{timestamp}"
    out_dir   = os.path.join(args.logdir, run_name)

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

    print(f"Creating {args.n_envs} envs (home side only, away handled by workers)...")

    train_env_raw = create_selfplay_vec_env(n_envs=args.n_envs, seed=args.seed, **env_kwargs)
    train_env     = VecMonitor(train_env_raw)

    # Eval env: single env with scripted opponent (never reassigned)
    eval_env_raw = create_selfplay_vec_env(n_envs=1, seed=args.seed + 10_000, **env_kwargs)
    eval_env     = VecMonitor(eval_env_raw)

    print(f"Observation space : {train_env.observation_space}")
    print(f"Action space      : {train_env.action_space}")
    print(f"Num envs (SB3)    : {train_env.num_envs}")

    # Device selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
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

    pool_cb = PoolCheckpointCallback(
        train_env=train_env_raw,
        ckpt_dir=ckpt_dir,
        bootstrap_steps=args.bootstrap_steps,
        checkpoint_freq=50_000,
        opponent_update_freq=args.opponent_update_freq,
        scripted_frac=args.scripted_frac,
        recent_frac=args.recent_frac,
        max_pool_size=args.max_pool_size,
        verbose=1,
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

    stats_cb  = ShootoutStatsCallback(rolling_window=100)
    callbacks = CallbackList([stats_cb, pool_cb, eval_cb])

    print(f"\nTraining for {args.total_timesteps:,} timesteps")
    print(f"Phase 0 (bootstrap): first {args.bootstrap_steps:,} steps → scripted opponent")
    print(f"Phase 1 (pool):      scripted={args.scripted_frac:.0%}  "
          f"recent={args.recent_frac:.0%}  "
          f"historical={1 - args.scripted_frac - args.recent_frac:.0%}")
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
