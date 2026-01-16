"""
Keyboard teleop for the single-goalie FoosballGoalieEnv.

Usage:
  python teleop_goalie.py --episodes 3

Controls (no W/A keys):
  - Up Arrow:    move slider up (positive y)
  - Down Arrow:  move slider down (negative y)
  - Right Arrow: kick forward (sets kicker to +1)
  - Left Arrow:  retract kicker (sets kicker to -1)
  - Q:           quit current episode

The script prints the reward each step so you can verify reward wiring.
"""

from __future__ import annotations

import argparse
from typing import Dict

import numpy as np
import pybullet as p

from FoosballGoalieEnv import FoosballGoalieEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teleop FoosballGoalieEnv and print rewards.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to play.")
    parser.add_argument("--action_repeat", type=int, default=8, help="Action repeat for the env.")
    parser.add_argument("--speed_min", type=float, default=2.0)
    parser.add_argument("--speed_max", type=float, default=10.0)
    parser.add_argument("--bounce_prob", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def teleop_action(control_state: Dict[str, float], step_size: float = 0.05) -> np.ndarray:
    """
    Map keyboard input to [slider, kicker] in [-1, 1].
    Uses arrow keys only to avoid reserved keys.
    """
    keys = p.getKeyboardEvents()

    if keys.get(ord("q"), 0) & p.KEY_WAS_TRIGGERED:
        raise KeyboardInterrupt

    delta = 0.0
    if keys.get(p.B3G_UP_ARROW, 0) & p.KEY_IS_DOWN:
        delta += step_size
    if keys.get(p.B3G_DOWN_ARROW, 0) & p.KEY_IS_DOWN:
        delta -= step_size
    control_state["slider"] = float(np.clip(control_state["slider"] + delta, -1.0, 1.0))

    if keys.get(p.B3G_RIGHT_ARROW, 0) & p.KEY_IS_DOWN:
        control_state["kicker"] = 1.0
    elif keys.get(p.B3G_LEFT_ARROW, 0) & p.KEY_IS_DOWN:
        control_state["kicker"] = -1.0
    else:
        # ease back toward neutral if no kicker key is held
        control_state["kicker"] = float(np.clip(control_state["kicker"] * 0.9, -1.0, 1.0))

    return np.array([control_state["slider"], control_state["kicker"]], dtype=np.float32)


def main() -> None:
    args = parse_args()

    env = FoosballGoalieEnv(
        render_mode="human",
        action_repeat=args.action_repeat,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
        bounce_prob=args.bounce_prob,
    )

    control_state = {"slider": 0.0, "kicker": -1.0}

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        ep_reward = 0.0
        step_idx = 0

        try:
            while not done:
                step_idx += 1
                try:
                    action = teleop_action(control_state)
                except KeyboardInterrupt:
                    print("Quit requested (Q). Ending episode early.")
                    break

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                ep_reward += float(reward)
                comps = info.get("reward_components")
                print(
                    f"[ep {ep:02d} step {step_idx:04d}] reward={float(reward):7.3f} "
                    f"event={info.get('event')} comps={comps}"
                )
        finally:
            print(f"Episode {ep} finished. Total reward: {ep_reward:.3f}")

    env.close()


if __name__ == "__main__":
    main()
