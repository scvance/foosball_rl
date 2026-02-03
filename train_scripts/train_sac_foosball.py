"""
Train SAC on FoosballGoalieEnv with TensorBoard logs + custom metrics for blocks/goals.

Run:
  python train_sac_foosball.py

Then:
  tensorboard --logdir .
"""

import os
import time
import argparse
from collections import deque
import multiprocessing as mp

import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed

from foosball_envs.FoosballGoalieEnv import FoosballGoalieEnv


class EventStatsCallback(BaseCallback):
    """
    Logs goal/block/out counts + rates to TensorBoard.

    Requires that env.step() info contains:
      - info["event"] in {"goal","block","out",None}
      - info["dense"] (optional, float)
    """

    def __init__(self, rolling_window: int = 100, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.rolling_window = int(rolling_window)

        self.goal_hist = deque(maxlen=self.rolling_window)
        self.block_hist = deque(maxlen=self.rolling_window)
        self.out_hist = deque(maxlen=self.rolling_window)

        self.goals_total = 0
        self.blocks_total = 0
        self.outs_total = 0
        self.episodes_total = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        dones = self.locals.get("dones", None)

        if infos is None or dones is None:
            return True

        for info, done in zip(infos, dones):
            if not done:
                continue

            self.episodes_total += 1
            event = info.get("event", None)

            is_goal = 1 if event == "goal" else 0
            is_block = 1 if event == "block" else 0
            is_out = 1 if event == "out" else 0

            self.goals_total += is_goal
            self.blocks_total += is_block
            self.outs_total += is_out

            self.goal_hist.append(is_goal)
            self.block_hist.append(is_block)
            self.out_hist.append(is_out)

            # per-episode flags
            self.logger.record("train/episode_is_goal", float(is_goal))
            self.logger.record("train/episode_is_block", float(is_block))
            self.logger.record("train/episode_is_out", float(is_out))

            # terminal dense
            if "dense" in info:
                self.logger.record("train/episode_terminal_dense", float(info["dense"]))

            # Optional telemetry if you included it in env.info
            if "slider_vel" in info:
                self.logger.record("train/episode_terminal_slider_vel", float(info["slider_vel"]))
            if "kicker_vel" in info:
                self.logger.record("train/episode_terminal_kicker_vel", float(info["kicker_vel"]))
            if "steps_per_policy" in info:
                self.logger.record("train/steps_per_policy", float(info["steps_per_policy"]))
            if "dt_sim" in info:
                self.logger.record("train/dt_sim", float(info["dt_sim"]))
            if "dt_policy" in info:
                self.logger.record("train/dt_policy", float(info["dt_policy"]))

        # cumulative + rolling rates
        if self.episodes_total > 0:
            self.logger.record("train/episodes_total", float(self.episodes_total))
            self.logger.record("train/goals_total", float(self.goals_total))
            self.logger.record("train/blocks_total", float(self.blocks_total))
            self.logger.record("train/outs_total", float(self.outs_total))

            self.logger.record("train/goal_rate_total", self.goals_total / self.episodes_total)
            self.logger.record("train/block_rate_total", self.blocks_total / self.episodes_total)
            self.logger.record("train/out_rate_total", self.outs_total / self.episodes_total)

        if len(self.goal_hist) > 0:
            self.logger.record("train/goal_rate_rolling", float(np.mean(self.goal_hist)))
            self.logger.record("train/block_rate_rolling", float(np.mean(self.block_hist)))
            self.logger.record("train/out_rate_rolling", float(np.mean(self.out_hist)))

        return True


def make_env(render_mode: str, seed: int, idx: int, **env_kwargs):
    def _init():
        env = FoosballGoalieEnv(render_mode=render_mode, seed=seed + idx, **env_kwargs)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default=".", help="TensorBoard log root (tensorboard --logdir . will work).")
    parser.add_argument("--run_name", type=str, default=None, help="Optional run name. Defaults to sac_foosball_<timestamp>.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--n_envs", type=int, default=max(1, min(12, mp.cpu_count())))

    parser.add_argument("--eval_freq", type=int, default=50_000)
    parser.add_argument("--n_eval_episodes", type=int, default=10)

    # Env params (shots)
    parser.add_argument("--speed_min", type=float, default=2.0)
    parser.add_argument("--speed_max", type=float, default=15.0)
    parser.add_argument("--bounce_prob", type=float, default=0.25)

    # NEW: control rates
    parser.add_argument("--policy_hz", type=float, default=200.0, help="Policy update frequency (Hz).")
    parser.add_argument("--sim_hz", type=int, default=1000, help="Simulation frequency (Hz). Try 2000.")

    # NEW: action caps (what policy can ask for)
    parser.add_argument("--slider_vel_cap_mps", type=float, default=15.0)
    parser.add_argument("--kicker_vel_cap_rads", type=float, default=170.0)

    # Physics
    parser.add_argument("--num_substeps", type=int, default=8, help="PyBullet substeps (lower=faster, less accurate).")

    # SAC device
    parser.add_argument("--device", type=str, default="cpu", help="Device for SAC (cpu, mps, cuda, auto).")

    args = parser.parse_args()

    set_random_seed(args.seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"sac_foosball_{timestamp}"

    out_dir = os.path.join(args.logdir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    tb_dir = os.path.join(out_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)

    ckpt_dir = os.path.join(out_dir, "checkpoints")
    best_dir = os.path.join(out_dir, "best")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    env_kwargs = dict(
        # rates
        policy_hz=args.policy_hz,
        sim_hz=args.sim_hz,
        # caps
        slider_vel_cap_mps=args.slider_vel_cap_mps,
        kicker_vel_cap_rads=args.kicker_vel_cap_rads,
        # physics
        num_substeps=args.num_substeps,
        # shot distribution
        speed_min=args.speed_min,
        speed_max=args.speed_max,
        bounce_prob=args.bounce_prob,
        # keep training headless
        real_time_gui=False,
    )

    # -------- Training env --------
    train_fns = [make_env("none", args.seed, i, **env_kwargs) for i in range(args.n_envs)]
    train_venv = SubprocVecEnv(train_fns) if args.n_envs > 1 else DummyVecEnv(train_fns)
    train_venv = VecMonitor(train_venv)

    # -------- Eval env --------
    eval_fns = [make_env("none", args.seed + 10_000, 0, **env_kwargs)]
    eval_env = DummyVecEnv(eval_fns)
    eval_env = VecMonitor(eval_env)

    # -------- Model --------
    model = SAC(
        policy="MlpPolicy",
        env=train_venv,
        verbose=0,
        tensorboard_log=tb_dir,
        device=args.device,
        learning_rate=3e-4,
        buffer_size=5_000_000,
        learning_starts=10_000,
        batch_size=64,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef="auto",
    )
    # model = SAC.load('/Users/samvance/vscode projects/Foosball_RL/sac_foosball_20260117_183329/checkpoints/sac_1500000_steps.zip', env=train_venv, device=args.device, buffer_size=5_000_000, tensorboard_log=tb_dir, learning_rate=5e-4, batch_size=256, train_freq=(1, "step"), gradient_steps=1, verbose=0)

    # -------- Callbacks --------
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=ckpt_dir,
        name_prefix="sac",
        save_replay_buffer=True,
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

    event_cb = EventStatsCallback(rolling_window=100)
    callbacks = CallbackList([event_cb, checkpoint_cb, eval_cb])

    # -------- Train --------
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name=run_name,
        progress_bar=True,
    )

    model.save(os.path.join(out_dir, "final_model"))

    train_venv.close()
    eval_env.close()

    print("\nDone.")
    print(f"Logs: {out_dir}")
    print("View with: tensorboard --logdir .")


if __name__ == "__main__":
    main()
