"""
Keyboard teleop to move the goalie slider and watch dense reward and intercept debug.

Controls:
  Up Arrow    : move slider +Y
  Down Arrow  : move slider -Y
  Q           : quit current episode

Notes:
  - Kicker is held retracted; slider velocity cap is kept high so position commands move quickly.
  - Prints dense reward each step so you can see alignment with the blue path box.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict

import numpy as np
import pybullet as p

from FoosballGoalieEnv import FoosballGoalieEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teleop goalie slider and view dense reward.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run.")
    parser.add_argument("--policy_hz", type=float, default=20.0, help="Policy rate for the env.")
    parser.add_argument("--sim_hz", type=int, default=1000, help="Sim rate for the env.")
    parser.add_argument("--step_size", type=float, default=0.15, help="Increment for slider command per key press.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for episodes.")
    return parser.parse_args()


def teleop_action(control_state: Dict[str, float], step_size: float) -> np.ndarray:
    """
    Map keyboard input to action [slider_pos, slider_vel, kicker_pos, kicker_vel].
    Only slider_pos is user-controlled; other fields are held constant.
    """
    keys = p.getKeyboardEvents()

    if keys.get(ord("q"), 0) & p.KEY_WAS_TRIGGERED:
        raise KeyboardInterrupt

    delta = 0.0
    if keys.get(p.B3G_UP_ARROW, 0) & p.KEY_IS_DOWN:
        delta += step_size
    if keys.get(p.B3G_DOWN_ARROW, 0) & p.KEY_IS_DOWN:
        delta -= step_size

    control_state["slider_pos"] = float(np.clip(control_state["slider_pos"] + delta, -1.0, 1.0))

    # Hold kicker near neutral (straight down) and keep velocity caps high for responsive motion.
    slider_vel_cmd = 1.0
    kicker_pos_cmd = control_state["kicker_neutral"]
    kicker_vel_cmd = 1.0

    return np.array(
        [control_state["slider_pos"], slider_vel_cmd, kicker_pos_cmd, kicker_vel_cmd],
        dtype=np.float32,
    )


def main() -> None:
    args = parse_args()

    env = FoosballGoalieEnv(render_mode="human", policy_hz=args.policy_hz, sim_hz=args.sim_hz, real_time_gui=True)

    # Precompute kicker neutral in normalized units: mid of limits -> [-1, 1].
    k_lo = env.sim.kicker_limits.lower
    k_hi = env.sim.kicker_limits.upper
    k_mid = 0.5 * (k_lo + k_hi)
    kicker_neutral = float((2.0 * (k_mid - k_lo) / max(k_hi - k_lo, 1e-6)) - 1.0)

    control_state = {"slider_pos": 0.0, "kicker_neutral": kicker_neutral}

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        step_idx = 0

        try:
            while not done:
                step_idx += 1
                try:
                    action = teleop_action(control_state, args.step_size)
                except KeyboardInterrupt:
                    print("Quit requested (Q). Ending episode early.")
                    break

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                dense = info.get("dense", 0.0)
                event = info.get("event")
                print(
                    f"[ep {ep:02d} step {step_idx:04d}] dense={dense:7.4f} reward={float(reward):7.4f} event={event}"
                )

                # Small sleep to avoid spinning too fast if render is off; with GUI on, env pacing handles this.
                if env.render_mode != "human":
                    time.sleep(1.0 / args.policy_hz)
        finally:
            print(f"Episode {ep} finished.")

    env.close()


if __name__ == "__main__":
    main()
