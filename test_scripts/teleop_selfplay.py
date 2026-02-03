"""
Teleop one foosball side against a trained SAC opponent and print that side's reward each step.

Usage (home controlled by keyboard, away uses SAC):
  python teleop_selfplay.py --control_side home --opponent_model selfplay_models/away_step2000000.zip

Controls:
  - W / S : move slider up / down (along the y-axis)
  - D or Space : kick forward (sets kicker target to +1)
  - A      : retract kicker (sets kicker target to -1)
  - Q      : quit episode early

Notes:
  - Action format for FoosballVersusEnv is [slider_pos, slider_vel, kicker_pos, kicker_vel] in [-1, 1].
  - The script prints the reward for the controlled side every step so you can verify reward wiring.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import pybullet as p
from stable_baselines3 import SAC

from foosball_envs.FoosballVersusEnv import FoosballVersusEnv


def _default_model_path(control_side: str) -> str | None:
    opponent_prefix = "away" if control_side == "home" else "home"
    candidate = os.path.join("selfplay_models", f"{opponent_prefix}_step2000000.zip")
    return candidate if os.path.exists(candidate) else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teleop one player vs SAC opponent and print rewards.")
    parser.add_argument("--control_side", choices=["home", "away"], default="home", help="Which side you control via keyboard.")
    parser.add_argument("--opponent_model", type=str, default=None, help="Path to the opponent SAC model zip.")
    parser.add_argument("--episodes", type=int, default=1, help="How many episodes to play.")
    parser.add_argument("--action_repeat", type=int, default=2, help="Passed to FoosballVersusEnv.")
    parser.add_argument("--serve_side", type=str, default="random", choices=["home", "away", "random"])
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def teleop_action(control_state: Dict[str, float], step_size: float = 0.05) -> np.ndarray:
    """
    Map keyboard input to an action vector [slider_pos, slider_vel, kicker_pos, kicker_vel].
    Slider position is incremented with W/S, kicker target toggles with A/D/Space.
    """
    keys = p.getKeyboardEvents()

    if keys.get(ord("q"), 0) & p.KEY_WAS_TRIGGERED:
        raise KeyboardInterrupt

    delta = 0.0
    if keys.get(ord("e"), 0) & p.KEY_IS_DOWN:
        delta += step_size
    if keys.get(ord("s"), 0) & p.KEY_IS_DOWN:
        delta -= step_size
    control_state["slider"] = float(np.clip(control_state["slider"] + delta, -1.0, 1.0))

    if (keys.get(ord("d"), 0) & p.KEY_IS_DOWN) or (keys.get(ord(" "), 0) & p.KEY_IS_DOWN):
        control_state["kicker"] = 1.0
    elif keys.get(ord("z"), 0) & p.KEY_IS_DOWN:
        control_state["kicker"] = -1.0
    else:
        # Relax toward neutral if no kicker input is held.
        control_state["kicker"] = float(np.clip(control_state["kicker"] * 0.95, -1.0, 1.0))

    slider_vel = 1.0
    kicker_vel = 1.0
    return np.array([control_state["slider"], slider_vel, control_state["kicker"], kicker_vel], dtype=np.float32)


def main() -> None:
    args = parse_args()

    opponent_model_path = args.opponent_model or _default_model_path(args.control_side)
    if opponent_model_path is None or not os.path.exists(opponent_model_path):
        raise FileNotFoundError(
            "--opponent_model is required; pass a SAC zip for the other side "
            f"(tried default {_default_model_path(args.control_side)})"
        )

    print(f"Controlling: {args.control_side} | Opponent model: {opponent_model_path}")
    env = FoosballVersusEnv(render_mode="human", action_repeat=args.action_repeat, serve_side=args.serve_side)
    opponent_model = SAC.load(opponent_model_path, device="auto")

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
                    human_action = teleop_action(control_state)
                except KeyboardInterrupt:
                    print("Quit requested (Q). Ending episode early.")
                    break

                if args.control_side == "home":
                    opp_action, _ = opponent_model.predict(obs["away"], deterministic=True)
                    action = {"home": human_action, "away": opp_action}
                else:
                    opp_action, _ = opponent_model.predict(obs["home"], deterministic=True)
                    action = {"home": opp_action, "away": human_action}

                obs, rewards, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                control_reward = float(rewards[args.control_side])
                ep_reward += control_reward

                comps = None
                if info.get("reward_components"):
                    comps = info["reward_components"].get(args.control_side)
                print(
                    f"[ep {ep:02d} step {step_idx:04d}] reward={control_reward:7.3f} "
                    f"event={info.get('event')} comps={comps}"
                )
        finally:
            print(f"Episode {ep} finished. Total reward for {args.control_side}: {ep_reward:.3f}")

    env.close()


if __name__ == "__main__":
    main()
