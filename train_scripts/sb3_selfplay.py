"""
Self-play SAC training with SB3 for Foosball.

One policy plays both sides. Observations and actions are mirrored in the env,
so the policy always sees the game from "home" perspective.

Key design:
- Policy sees single-agent obs (21 dims) and action (3 dims) spaces
- SB3 sees 2*n_underlying_envs virtual environments:
  - Even indices (0, 2, 4, ...) = home sides
  - Odd indices (1, 3, 5, ...) = away sides
- Policy steps once for home obs, once for away obs
- step_async pairs actions: (action[0], action[1]) sent to env 0, etc.
- Logging is per underlying env (not per virtual env) to avoid duplication

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
    
    Returns dict observations and rewards matching the underlying env.
    Info dict contains event, rewards, etc. for logging.
    """
    parent_remote.close()
    env = env_fn()

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

            # Include rewards in info for logging
            info["reward_home"] = reward_dict["home"]
            info["reward_away"] = reward_dict["away"]

            # Auto-reset on done
            new_obs_dict = None
            if done:
                info["terminal_observation_home"] = obs_dict["home"].copy()
                info["terminal_observation_away"] = obs_dict["away"].copy()
                new_obs_dict, reset_info = env.reset()

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
            attr_name = data
            val = getattr(env, attr_name, None)
            remote.send(val)

        else:
            raise NotImplementedError(f"Unknown command: {cmd}")


class SelfPlayVecEnv(VecEnv):
    """
    Vectorized self-play environment.

    Each underlying FoosballVersusEnv runs in its own subprocess.
    SB3 sees 2*n_underlying_envs virtual environments:
    - Virtual envs 0, 2, 4, ... are home sides
    - Virtual envs 1, 3, 5, ... are away sides
    
    On each step:
    1. SB3 provides 2*n actions (policy stepped once per virtual env)
    2. We pair them: (action[0], action[1]) for underlying env 0, etc.
    3. We return observations and rewards for both sides
    
    Info is returned per underlying env (attached to home virtual env only)
    to avoid duplicate logging.
    """

    def __init__(self, env_fns: List):
        self.n_underlying_envs = len(env_fns)
        self.waiting = False
        self.closed = False

        # Create pipes and processes
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.n_underlying_envs)])
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

        # SB3 sees 2x the number of underlying envs
        n_virtual_envs = 2 * self.n_underlying_envs
        super().__init__(n_virtual_envs, obs_space, act_space)

    def reset(self) -> np.ndarray:
        """Reset all envs and return observations from both sides."""
        for remote in self.remotes:
            remote.send(("reset", None))

        results = [remote.recv() for remote in self.remotes]

        obs = np.zeros((self.num_envs,) + self.observation_space.shape, dtype=np.float32)
        for i, (home_obs, away_obs, info) in enumerate(results):
            obs[2 * i] = home_obs      # Even indices = home
            obs[2 * i + 1] = away_obs  # Odd indices = away

        return obs

    def step_async(self, actions: np.ndarray) -> None:
        """
        Send actions to workers.
        
        Actions are paired: action[2*i] is home, action[2*i+1] is away
        for underlying env i.
        """
        self.waiting = True

        for i, remote in enumerate(self.remotes):
            home_action = actions[2 * i]
            away_action = actions[2 * i + 1]
            remote.send(("step", (home_action, away_action)))

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Wait for results from workers.
        
        Returns observations and rewards for all virtual envs.
        Info dicts contain full env info only for home (even) indices
        to avoid duplicate logging. Away (odd) indices get minimal info.
        """
        self.waiting = False

        results = [remote.recv() for remote in self.remotes]

        obs = np.zeros((self.num_envs,) + self.observation_space.shape, dtype=np.float32)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: List[Dict] = []

        for i, (home_obs, away_obs, home_reward, away_reward, done, info, new_home_obs, new_away_obs) in enumerate(results):
            # Home side (even index) - gets full info for logging
            home_info = info.copy()
            home_info["is_home"] = True
            if done and new_home_obs is not None:
                home_info["terminal_observation"] = info["terminal_observation_home"]
                obs[2 * i] = new_home_obs
            else:
                obs[2 * i] = home_obs
            rewards[2 * i] = home_reward
            dones[2 * i] = done
            infos.append(home_info)

            # Away side (odd index) - minimal info to avoid duplicate logging
            away_info = {"is_home": False}
            if done and new_away_obs is not None:
                away_info["terminal_observation"] = info["terminal_observation_away"]
                obs[2 * i + 1] = new_away_obs
            else:
                obs[2 * i + 1] = away_obs
            rewards[2 * i + 1] = away_reward
            dones[2 * i + 1] = done
            infos.append(away_info)

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
) -> SelfPlayVecEnv:
    """
    Create a SelfPlayVecEnv with n_envs underlying FoosballVersusEnvs.
    
    Note: SB3 will see 2*n_envs virtual environments.
    """
    env_fns = [EnvFactory(seed, i, env_kwargs) for i in range(n_envs)]
    return SelfPlayVecEnv(env_fns)


class SelfPlayStatsCallback(BaseCallback):
    """
    Logs self-play statistics to TensorBoard.
    
    Only processes info from home (even) indices to avoid duplicate counting,
    since each underlying env step produces one info dict attached to the
    home virtual env.
    """

    def __init__(self, rolling_window: int = 100, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.rolling_window = rolling_window

        self.home_goals = deque(maxlen=rolling_window)
        self.away_goals = deque(maxlen=rolling_window)
        self.home_blocks = deque(maxlen=rolling_window)
        self.away_blocks = deque(maxlen=rolling_window)
        self.outs = deque(maxlen=rolling_window)
        
        # Reward tracking
        self.home_rewards = deque(maxlen=rolling_window * 100)  # More samples for rewards
        self.away_rewards = deque(maxlen=rolling_window * 100)

        self.total_home_goals = 0
        self.total_away_goals = 0
        self.total_home_blocks = 0
        self.total_away_blocks = 0
        self.total_outs = 0
        self.total_episodes = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            # Only process home (even) indices to avoid duplicate logging
            if not info.get("is_home", False):
                continue
            
            # Log rewards per step
            if "reward_home" in info:
                self.home_rewards.append(info["reward_home"])
                self.away_rewards.append(info["reward_away"])
                
            # Log events (episode termination)
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

        # Log to tensorboard
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
        
        # Log reward stats
        if len(self.home_rewards) > 0:
            self.logger.record("selfplay/mean_home_reward", np.mean(self.home_rewards))
            self.logger.record("selfplay/mean_away_reward", np.mean(self.away_rewards))

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
    parser.add_argument("--max_episode_steps", type=int, default=5000)
    parser.add_argument("--serve_side", type=str, default="random")

    # Physics
    parser.add_argument("--speed_min", type=float, default=1.0)
    parser.add_argument("--speed_max", type=float, default=15.0)
    parser.add_argument("--bounce_prob", type=float, default=0.25)
    parser.add_argument("--num_substeps", type=int, default=8)

    # SAC params
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--buffer_size", type=int, default=100_000)
    parser.add_argument("--learning_starts", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint to load.")

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
        spawn_without_velocity=True,
    )

    print(f"Creating {args.n_envs} underlying envs ({2*args.n_envs} virtual envs for SB3)...")

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
    print(f"Num virtual envs (SB3 sees): {train_env.num_envs}")
    print(f"Num underlying envs: {args.n_envs}")

    # Select device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "cpu"  # MPS often slower for small networks
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Create SAC model
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
        print(f"Loading model from checkpoint: {args.ckpt}...")
        model = SAC.load(args.ckpt, env=train_env, device=device)
        print("Model loaded!")

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