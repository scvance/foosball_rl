"""
watch_shootout_env.py

Watch ShootoutVersusEnv with configurable policies for home and away.
Supports scripted-vs-scripted, model-vs-scripted, and self-play.

Examples
--------
Scripted vs scripted:
  python3 train_scripts/watch_shootout_env.py --home scripted --away scripted

Model vs scripted:
  python3 train_scripts/watch_shootout_env.py \
    --home /path/to/model.zip --away scripted --deterministic

Self-play with one model:
  python3 train_scripts/watch_shootout_env.py \
    --home /path/to/model.zip --away same --deterministic
"""

import argparse
import time
from typing import Any, Dict

import numpy as np
import pybullet as p
from stable_baselines3 import SAC

from foosball_envs.ShootoutVersusEnv import ShootoutVersusEnv
from train_scripts.train_shootout_versus import ScriptedPolicy


def _make_policy(spec: str, home_policy: Any = None):
    """
    spec values:
      - scripted
      - same      (reuse home policy object)
      - /path/to/model.zip
    """
    if spec == "scripted":
        return ScriptedPolicy(n_envs=1)
    if spec == "same":
        if home_policy is None:
            raise ValueError("--away same requires a loaded --home policy")
        return home_policy
    return SAC.load(spec, device="auto")


def _policy_action(policy: Any, obs: np.ndarray, deterministic: bool) -> np.ndarray:
    action, _ = policy.predict(obs, deterministic=deterministic)
    return np.asarray(action, dtype=np.float32)


def _start_episode(policy: Any):
    if hasattr(policy, "on_episode_start"):
        policy.on_episode_start(0)


def _obs_mirror_error(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Returns absolute error vector between actual away obs and expected mirrored-away obs
    computed from home obs under current ShootoutVersusEnv convention.
    """
    h = obs["home"]
    a = obs["away"]

    exp = np.zeros_like(h)

    # Ball est pos/vel/pred (x,y mirrored; z unchanged)
    exp[0] = -h[0]
    exp[1] = -h[1]
    exp[2] = h[2]

    exp[3] = -h[3]
    exp[4] = -h[4]
    exp[5] = h[5]

    exp[6] = -h[6]
    exp[7] = -h[7]
    exp[8] = h[8]

    # own (away) from home's opp, and opp (home) from home's own
    # Layout: [paddle_ang, handle_pos, paddle_vel, handle_vel]
    exp[9] = h[13]
    exp[10] = -h[14]
    exp[11] = h[15]
    exp[12] = -h[16]

    exp[13] = h[9]
    exp[14] = -h[10]
    exp[15] = h[11]
    exp[16] = -h[12]

    # Intercept [y, z, x_plane, t]
    exp[17] = -h[17]
    exp[18] = h[18]
    exp[19] = -h[19]
    exp[20] = h[20]

    return np.abs(a - exp)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--home", type=str, default="scripted",
                   help="Home policy: 'scripted' or path to .zip")
    p.add_argument("--away", type=str, default="scripted",
                   help="Away policy: 'scripted', 'same', or path to .zip")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true",
                   help="Use deterministic actions for SAC policies")

    p.add_argument("--verify_mirroring", action="store_true",
                   help="Print observation-mirroring residuals each episode")

    # Keep aligned with training defaults unless intentionally changed.
    p.add_argument("--policy_hz", type=float, default=30.0)
    p.add_argument("--sim_hz", type=int, default=240)
    p.add_argument("--max_episode_steps", type=int, default=200)
    p.add_argument("--serve_mode", type=str, default="random",
                   choices=["random_fire", "corner", "random"])
    p.add_argument("--handle_vel_cap_mps", type=float, default=10.0)
    p.add_argument("--paddle_vel_cap_rads", type=float, default=20.0)
    p.add_argument("--ball_restitution", type=float, default=0.30)
    p.add_argument("--wall_restitution", type=float, default=0.85)
    p.add_argument("--paddle_restitution", type=float, default=0.85)
    p.add_argument("--num_substeps", type=int, default=1)
    p.add_argument(
        "--no_reward_overlay",
        action="store_true",
        help="Disable per-step and cumulative reward 3D text near each goal.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    env = ShootoutVersusEnv(
        render_mode="human",
        policy_hz=args.policy_hz,
        sim_hz=args.sim_hz,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
        serve_mode=args.serve_mode,
        handle_vel_cap_mps=args.handle_vel_cap_mps,
        paddle_vel_cap_rads=args.paddle_vel_cap_rads,
        ball_restitution=args.ball_restitution,
        wall_restitution=args.wall_restitution,
        paddle_restitution=args.paddle_restitution,
        num_substeps=args.num_substeps,
        real_time_gui=True,
    )

    home_policy = _make_policy(args.home)
    away_policy = _make_policy(args.away, home_policy=home_policy)

    print(f"Home policy: {args.home}")
    print(f"Away policy: {args.away}")

    sim = env.sim
    reward_text_home_id = None
    reward_text_away_id = None
    z_text = float(sim.floor_z_edge + 0.05)  # right above the table near the goal lines
    home_anchor_world = sim.table_local_to_world_pos([-sim.goal_x, 0.0, z_text])
    away_anchor_world = sim.table_local_to_world_pos([sim.goal_x, 0.0, z_text])

    def update_reward_overlay(step_home: float, step_away: float, ret_home: float, ret_away: float) -> None:
        nonlocal reward_text_home_id, reward_text_away_id
        if args.no_reward_overlay:
            return

        home_text = f"home r={step_home:+.3f}\\nR={ret_home:+.3f}"
        away_text = f"away r={step_away:+.3f}\\nR={ret_away:+.3f}"

        kwargs_home = {}
        kwargs_away = {}
        if reward_text_home_id is not None:
            kwargs_home["replaceItemUniqueId"] = reward_text_home_id
        if reward_text_away_id is not None:
            kwargs_away["replaceItemUniqueId"] = reward_text_away_id

        reward_text_home_id = p.addUserDebugText(
            home_text,
            home_anchor_world,
            textColorRGB=[0.95, 0.25, 0.25],
            textSize=1.1,
            lifeTime=0,
            **kwargs_home,
        )
        reward_text_away_id = p.addUserDebugText(
            away_text,
            away_anchor_world,
            textColorRGB=[0.25, 0.90, 0.35],
            textSize=1.1,
            lifeTime=0,
            **kwargs_away,
        )

    try:
        for ep in range(1, args.episodes + 1):
            obs, info = env.reset()
            done = False
            steps = 0
            home_ret = 0.0
            away_ret = 0.0

            # Avoid double-calling when both sides share same object.
            if away_policy is home_policy:
                _start_episode(home_policy)
            else:
                _start_episode(home_policy)
                _start_episode(away_policy)

            mirror_err_mean = []
            mirror_err_max = []
            update_reward_overlay(0.0, 0.0, home_ret, away_ret)

            while not done:
                h_action = _policy_action(home_policy, obs["home"], args.deterministic)
                a_action = _policy_action(away_policy, obs["away"], args.deterministic)

                obs, rewards, terminated, truncated, info = env.step(
                    {"home": h_action, "away": a_action}
                )

                if args.verify_mirroring:
                    err = _obs_mirror_error(obs)
                    mirror_err_mean.append(float(np.mean(err)))
                    mirror_err_max.append(float(np.max(err)))

                home_ret += float(rewards["home"])
                away_ret += float(rewards["away"])
                update_reward_overlay(
                    float(rewards["home"]),
                    float(rewards["away"]),
                    home_ret,
                    away_ret,
                )
                steps += 1
                done = bool(terminated or truncated)

            line = (
                f"Episode {ep}/{args.episodes} | steps={steps} | "
                f"event={info.get('event')} | home_reward={home_ret:.3f} | away_reward={away_ret:.3f}"
            )
            if args.verify_mirroring and mirror_err_mean:
                line += (
                    f" | mirror_err_mean={np.mean(mirror_err_mean):.3e}"
                    f" | mirror_err_max={np.max(mirror_err_max):.3e}"
                )
            print(line)

            time.sleep(0.25)
    finally:
        env.close()


if __name__ == "__main__":
    main()
