"""
Self-play SAC training with SB3 for Foosball.

One policy plays both sides. Observations and actions are mirrored in the env,
so the policy always sees the game from "home" perspective.

Key design:
- Policy sees single-agent obs (21 dims) and action (3 dims) spaces
- Each underlying env step, we only train the DEFENDING side
- This eliminates noise from the non-defending side's experiences
- SB3 sees n_envs environments (one per underlying env)

Uses multiprocessing for parallel environment execution.
"""

import os
import time
import argparse
from collections import deque
import multiprocessing as mp
from multiprocessing import Process, Pipe
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed

from gymnasium import spaces

from foosball_envs.FoosballVersusEnv import FoosballVersusEnv


def _worker(remote, parent_remote, env_fn):
    """
    Worker process that runs a single FoosballVersusEnv.

    Key insight: We determine which side is defending based on ball velocity,
    and only return that side's experience. The other side's action is still
    applied (for physics), but we don't train on it.
    """
    parent_remote.close()
    env = env_fn()

    # Track current defending side
    defending_side = "home"  # Will be updated on reset

    def get_defending_side():
        """Determine which side is defending based on ball velocity."""
        try:
            ball_vel = env._est_vel
            vx = float(ball_vel[0])
            # Ball moving left (negative x) -> heading to home goal -> home defends
            # Ball moving right (positive x) -> heading to away goal -> away defends
            if vx < -0.1:
                return "home"
            elif vx > 0.1:
                return "away"
            else:
                # Ball nearly stationary, keep current
                return defending_side
        except:
            return "home"

    while True:
        try:
            cmd, data = remote.recv()
        except EOFError:
            break

        if cmd == "step":
            home_action, away_action = data
            action_dict = {"home": home_action, "away": away_action}
            obs_dict, reward_dict, terminated, truncated, info = env.step(action_dict)
            done = terminated or truncated

            # Determine which side was defending this step
            # Use the side we identified at the start of step
            current_defending = defending_side

            # Update defending side for next step
            defending_side = get_defending_side()

            # Get the defending side's data
            if current_defending == "home":
                obs = obs_dict["home"]
                reward = reward_dict["home"]
                action_used = home_action
            else:
                obs = obs_dict["away"]
                reward = reward_dict["away"]
                action_used = away_action

            info["defending_side"] = current_defending
            info["reward_home"] = reward_dict["home"]
            info["reward_away"] = reward_dict["away"]

            # Auto-reset on done
            if done:
                info["terminal_observation"] = obs.copy()
                obs_dict_new, reset_info = env.reset()
                # After reset, determine new defending side
                defending_side = get_defending_side()
                # Return the new defending side's observation
                if defending_side == "home":
                    obs = obs_dict_new["home"]
                else:
                    obs = obs_dict_new["away"]

            remote.send((obs, reward, done, info, defending_side))

        elif cmd == "reset":
            obs_dict, info = env.reset()
            # Determine initial defending side
            defending_side = get_defending_side()
            if defending_side == "home":
                obs = obs_dict["home"]
            else:
                obs = obs_dict["away"]
            remote.send((obs, info, defending_side))

        elif cmd == "close":
            remote.close()
            break

        elif cmd == "get_spaces":
            remote.send((env.observation_space, env.action_space))

        elif cmd == "get_attr":
            attr_name = data
            val = getattr(env, attr_name, None)
            remote.send(val)

        else:
            raise NotImplementedError(f"Unknown command: {cmd}")


class SelfPlayDefenderVecEnv(VecEnv):
    """
    Vectorized self-play environment that only trains the defending side.

    Each underlying FoosballVersusEnv runs in its own subprocess.
    On each step:
    1. SB3 provides n actions (one per env)
    2. We use each action for BOTH home and away in that env
    3. We return only the DEFENDING side's observation and reward
    4. This ensures 100% of experiences are relevant

    The key insight: since the policy is mirrored, using the same action
    for both sides is equivalent to having two independent policies that
    happen to be identical.
    """

    def __init__(self, env_fns: List):
        self.n_envs = len(env_fns)
        self.waiting = False
        self.closed = False

        # Track which side is defending in each env
        self._defending_sides: List[str] = ["home"] * self.n_envs

        # Create pipes and processes
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.n_envs)])
        self.processes = []

        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            process = Process(target=_worker, args=(work_remote, remote, env_fn), daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        # Get spaces from first env
        self.remotes[0].send(("get_spaces", None))
        obs_space_dict, act_space_dict = self.remotes[0].recv()

        # Single-agent spaces (home and away are identical due to mirroring)
        obs_space = obs_space_dict["home"]
        act_space = act_space_dict["home"]

        super().__init__(self.n_envs, obs_space, act_space)

    def reset(self) -> np.ndarray:
        """Reset all envs and return observations from defending sides."""
        for remote in self.remotes:
            remote.send(("reset", None))

        results = [remote.recv() for remote in self.remotes]

        obs = np.zeros((self.n_envs,) + self.observation_space.shape, dtype=np.float32)
        for i, (ob, info, defending_side) in enumerate(results):
            obs[i] = ob
            self._defending_sides[i] = defending_side

        return obs

    def step_async(self, actions: np.ndarray) -> None:
        """Send actions to workers. Each action is used for BOTH sides."""
        self.waiting = True

        for i, remote in enumerate(self.remotes):
            # Use the same action for both home and away
            # The observation is already from the defending side's perspective
            action = actions[i]
            remote.send(("step", (action, action)))

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Wait for results from workers."""
        self.waiting = False

        results = [remote.recv() for remote in self.remotes]

        obs = np.zeros((self.n_envs,) + self.observation_space.shape, dtype=np.float32)
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=bool)
        infos: List[Dict] = []

        for i, (ob, reward, done, info, defending_side) in enumerate(results):
            obs[i] = ob
            rewards[i] = reward
            dones[i] = done
            infos.append(info)
            self._defending_sides[i] = defending_side

        return obs, rewards, dones, infos

    def close(self) -> None:
        """Close all workers."""
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def env_is_wrapped(self, wrapper_class, indices=None) -> List[bool]:
        return [False] * self.num_envs

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        return [None] * self.num_envs

    def get_attr(self, attr_name: str, indices=None):
        self.remotes[0].send(("get_attr", attr_name))
        val = self.remotes[0].recv()
        return [val] * self.num_envs

    def set_attr(self, attr_name: str, value, indices=None):
        pass

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        return [None] * self.num_envs


class EnvFactory:
    """Picklable factory for creating FoosballVersusEnv instances."""
    def __init__(self, seed: int, idx: int, env_kwargs: Dict):
        self.seed = seed
        self.idx = idx
        self.env_kwargs = env_kwargs

    def __call__(self) -> FoosballVersusEnv:
        return FoosballVersusEnv(seed=self.seed + self.idx, **self.env_kwargs)


def create_selfplay_vec_env(
    n_envs: int,
    seed: int,
    **env_kwargs,
) -> SelfPlayDefenderVecEnv:
    """Create a SelfPlayDefenderVecEnv with n_envs underlying FoosballVersusEnvs."""
    env_fns = [EnvFactory(seed, i, env_kwargs) for i in range(n_envs)]
    return SelfPlayDefenderVecEnv(env_fns)


class SelfPlayStatsCallback(BaseCallback):
    """
    Logs self-play statistics to TensorBoard.
    Tracks goals, blocks, and win rates.
    """

    def __init__(self, rolling_window: int = 100, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.rolling_window = rolling_window

        self.home_goals = deque(maxlen=rolling_window)
        self.away_goals = deque(maxlen=rolling_window)
        self.home_blocks = deque(maxlen=rolling_window)
        self.away_blocks = deque(maxlen=rolling_window)
        self.outs = deque(maxlen=rolling_window)

        self.total_home_goals = 0
        self.total_away_goals = 0
        self.total_home_blocks = 0
        self.total_away_blocks = 0
        self.total_outs = 0
        self.total_episodes = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            event = info.get("event")
            if event is None:
                continue

            self.total_episodes += 1

            is_home_goal = 1 if event == "home_goal" else 0
            is_away_goal = 1 if event == "away_goal" else 0
            is_home_block = 1 if event == "home_block" else 0
            is_away_block = 1 if event == "away_block" else 0
            is_out = 1 if event in ("out", "stalled") else 0

            self.home_goals.append(is_home_goal)
            self.away_goals.append(is_away_goal)
            self.home_blocks.append(is_home_block)
            self.away_blocks.append(is_away_block)
            self.outs.append(is_out)

            self.total_home_goals += is_home_goal
            self.total_away_goals += is_away_goal
            self.total_home_blocks += is_home_block
            self.total_away_blocks += is_away_block
            self.total_outs += is_out

        if self.total_episodes > 0:
            self.logger.record("selfplay/total_episodes", self.total_episodes)
            self.logger.record("selfplay/home_goals_total", self.total_home_goals)
            self.logger.record("selfplay/away_goals_total", self.total_away_goals)
            self.logger.record("selfplay/home_blocks_total", self.total_home_blocks)
            self.logger.record("selfplay/away_blocks_total", self.total_away_blocks)
            self.logger.record("selfplay/outs_total", self.total_outs)

            if len(self.home_goals) > 0:
                block_rate = np.mean(self.home_blocks) + np.mean(self.away_blocks)
                goal_rate = np.mean(self.home_goals) + np.mean(self.away_goals)
                self.logger.record("selfplay/block_rate", block_rate)
                self.logger.record("selfplay/goal_rate", goal_rate)
                self.logger.record("selfplay/home_goal_rate", np.mean(self.home_goals))
                self.logger.record("selfplay/away_goal_rate", np.mean(self.away_goals))
                self.logger.record("selfplay/home_block_rate", np.mean(self.home_blocks))
                self.logger.record("selfplay/away_block_rate", np.mean(self.away_blocks))
                self.logger.record("selfplay/out_rate", np.mean(self.outs))

        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default=".", help="TensorBoard log root.")
    parser.add_argument("--run_name", type=str, default=None, help="Optional run name.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--n_envs", type=int, default=max(1, min(12, mp.cpu_count())))

    parser.add_argument("--eval_freq", type=int, default=50_000)
    parser.add_argument("--n_eval_episodes", type=int, default=10)

    # Env params
    parser.add_argument("--policy_hz", type=float, default=200.0)
    parser.add_argument("--sim_hz", type=int, default=1000)
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--serve_side", type=str, default="random")

    # Physics
    parser.add_argument("--speed_min", type=float, default=2.0)
    parser.add_argument("--speed_max", type=float, default=15.0)
    parser.add_argument("--bounce_prob", type=float, default=0.25)
    parser.add_argument("--num_substeps", type=int, default=8)

    # SAC params
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--buffer_size", type=int, default=1_000_000)
    parser.add_argument("--learning_starts", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)

    args = parser.parse_args()

    set_random_seed(args.seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"sac_selfplay_{timestamp}"

    out_dir = os.path.join(args.logdir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    tb_dir = os.path.join(out_dir, "tb")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    best_dir = os.path.join(out_dir, "best")
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    env_kwargs = dict(
        render_mode="none",
        policy_hz=args.policy_hz,
        sim_hz=args.sim_hz,
        max_episode_steps=args.max_episode_steps,
        serve_side=args.serve_side,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
        bounce_prob=args.bounce_prob,
        num_substeps=args.num_substeps,
        real_time_gui=False,
    )

    print(f"Creating {args.n_envs} envs (training ONLY defending side each step)...")

    # Create self-play vectorized environment
    train_env = create_selfplay_vec_env(
        n_envs=args.n_envs,
        seed=args.seed,
        **env_kwargs,
    )
    train_env = VecMonitor(train_env)

    # Create eval env
    eval_env = create_selfplay_vec_env(
        n_envs=1,
        seed=args.seed + 10000,
        **env_kwargs,
    )
    eval_env = VecMonitor(eval_env)

    print(f"Observation space: {train_env.observation_space}")
    print(f"Action space: {train_env.action_space}")
    print(f"Num envs: {train_env.num_envs}")

    # Select device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "cpu"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Create SAC model
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

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=ckpt_dir,
        name_prefix="sac_selfplay",
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

    stats_cb = SelfPlayStatsCallback(rolling_window=100)

    callbacks = CallbackList([stats_cb, checkpoint_cb, eval_cb])

    print(f"\nStarting training for {args.total_timesteps} timesteps...")
    print(f"Logs: {out_dir}")
    print("View with: tensorboard --logdir .\n")

    # Train
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name=run_name,
        progress_bar=True,
    )

    # Save final model
    model.save(os.path.join(out_dir, "final_model"))

    train_env.close()
    eval_env.close()

    print("\nTraining complete!")
    print(f"Final model saved to: {os.path.join(out_dir, 'final_model')}")


if __name__ == "__main__":
    main()
