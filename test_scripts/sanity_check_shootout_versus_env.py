"""
Sanity checks for ShootoutVersusEnv.

Checks:
1) Action/observation space structure and shapes
2) reset()/step() API contract and output validity
3) Observation mirroring consistency (home vs away)
4) Action un-mirroring behavior for away side

Usage:
  python test_scripts/sanity_check_shootout_versus_env.py
  python test_scripts/sanity_check_shootout_versus_env.py --render
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from foosball_envs.ShootoutVersusEnv import ShootoutVersusEnv


def header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def expected_away_from_home(home_obs: np.ndarray) -> np.ndarray:
    """Expected mirrored-away observation from the home observation layout."""
    exp = np.zeros_like(home_obs, dtype=np.float32)

    # Ball est pos/vel/pred: mirror x,y and keep z
    exp[0] = -home_obs[0]
    exp[1] = -home_obs[1]
    exp[2] = home_obs[2]

    exp[3] = -home_obs[3]
    exp[4] = -home_obs[4]
    exp[5] = home_obs[5]

    exp[6] = -home_obs[6]
    exp[7] = -home_obs[7]
    exp[8] = home_obs[8]

    # own/opp swap with handle pos/vel mirrored (negated)
    # [paddle_ang, handle_pos, paddle_vel, handle_vel]
    exp[9] = home_obs[13]
    exp[10] = -home_obs[14]
    exp[11] = home_obs[15]
    exp[12] = -home_obs[16]

    exp[13] = home_obs[9]
    exp[14] = -home_obs[10]
    exp[15] = home_obs[11]
    exp[16] = -home_obs[12]

    # Intercept terms are side-specific predictions and are not guaranteed to be
    # pure mirrors of the other side's values; skip strict equality checks there.
    exp[17:21] = np.nan

    return exp


def angle_abs_err(a: float, b: float) -> float:
    """Smallest absolute angular distance accounting for wrap at +/-pi."""
    d = ((a - b + np.pi) % (2.0 * np.pi)) - np.pi
    return abs(float(d))


def test_spaces(env: ShootoutVersusEnv) -> None:
    header("TEST 1: Space Definitions")

    check(set(env.action_space.keys()) == {"home", "away"}, "Action keys must be home/away")
    check(set(env.observation_space.keys()) == {"home", "away"}, "Obs keys must be home/away")

    for side in ("home", "away"):
        act_space = env.action_space[side]
        obs_space = env.observation_space[side]
        check(act_space.shape == (3,), f"{side} action shape must be (3,)")
        check(obs_space.shape == (21,), f"{side} obs shape must be (21,)")
        check(act_space.dtype == np.float32, f"{side} action dtype must be float32")
        check(obs_space.dtype == np.float32, f"{side} obs dtype must be float32")

    print("[PASS] Action and observation spaces match expected ShootoutVersusEnv layout")


def test_reset_step_contract(env: ShootoutVersusEnv, steps: int) -> None:
    header("TEST 2: reset()/step() Contract")

    obs, info = env.reset()
    check(isinstance(obs, dict), "reset() obs must be dict")
    check(isinstance(info, dict), "reset() info must be dict")
    check("shot" in info, "reset() info should include shot metadata")

    for side in ("home", "away"):
        check(obs[side].shape == (21,), f"reset obs {side} shape mismatch")
        check(obs[side].dtype == np.float32, f"reset obs {side} dtype mismatch")

    n_term = 0
    n_trunc = 0

    for i in range(steps):
        if i % 2 == 0:
            action = {
                "home": np.zeros(3, dtype=np.float32),
                "away": np.zeros(3, dtype=np.float32),
            }
        else:
            action = {
                "home": env.action_space["home"].sample(),
                "away": env.action_space["away"].sample(),
            }

        obs, rewards, terminated, truncated, info = env.step(action)

        check(isinstance(obs, dict), "step() obs must be dict")
        check(isinstance(rewards, dict), "step() rewards must be dict")
        check(isinstance(terminated, bool), "terminated must be bool")
        check(isinstance(truncated, bool), "truncated must be bool")
        check(isinstance(info, dict), "step() info must be dict")

        check(set(rewards.keys()) == {"home", "away"}, "reward keys must be home/away")
        check(np.isfinite(float(rewards["home"])), "home reward must be finite")
        check(np.isfinite(float(rewards["away"])), "away reward must be finite")

        for side in ("home", "away"):
            check(obs[side].shape == (21,), f"step obs {side} shape mismatch")
            check(obs[side].dtype == np.float32, f"step obs {side} dtype mismatch")

        if terminated:
            n_term += 1
        if truncated:
            n_trunc += 1

        if terminated or truncated:
            obs, _ = env.reset()

    print(f"[PASS] Step contract validated for {steps} steps (terminated={n_term}, truncated={n_trunc})")


def test_observation_mirroring(env: ShootoutVersusEnv, trials: int, tol: float) -> None:
    header("TEST 3: Observation Mirroring")

    mean_errors = []
    max_errors = []

    for _ in range(trials):
        obs, _ = env.reset()

        for _ in range(6):
            a = env.action_space["home"].sample()
            b = env.action_space["away"].sample()
            obs, _, terminated, truncated, _ = env.step({"home": a, "away": b})
            if terminated or truncated:
                obs, _ = env.reset()

        expected = expected_away_from_home(obs["home"])
        err = np.abs(obs["away"] - expected)
        err[np.isnan(expected)] = 0.0

        # Paddle-angle indices can differ by +/- 2*pi across wrap boundaries.
        err[9] = angle_abs_err(float(obs["away"][9]), float(expected[9]))
        err[13] = angle_abs_err(float(obs["away"][13]), float(expected[13]))
        mean_errors.append(float(np.mean(err)))
        max_errors.append(float(np.max(err)))

    worst = float(np.max(max_errors)) if max_errors else 0.0
    mean = float(np.mean(mean_errors)) if mean_errors else 0.0

    print(f"mirror mean error: {mean:.3e}")
    print(f"mirror max  error: {worst:.3e}")
    check(worst < tol, f"Mirroring residual too high: {worst:.3e} >= {tol:.3e}")
    print("[PASS] Observation mirroring is consistent")


def test_action_unmirroring(env: ShootoutVersusEnv, hold_steps: int) -> None:
    header("TEST 4: Away Action Un-Mirroring")

    env.reset()

    # Same normalized handle position for both sides should produce mirrored handle positions.
    action = {
        "home": np.array([0.7, 1.0, 0.0], dtype=np.float32),
        "away": np.array([0.7, 1.0, 0.0], dtype=np.float32),
    }
    for _ in range(hold_steps):
        env.step(action)

    h_home, _ = env.sim.get_joint_positions()
    h_away, _ = env.sim.get_opponent_joint_positions()

    print(f"handle_home={h_home:+.4f}, handle_away={h_away:+.4f}, sum={h_home + h_away:+.4e}")
    check(abs(h_home + h_away) < 2.0e-2, "Away handle action should be un-mirrored (negated target)")

    # Same paddle velocity command should preserve sign for both sides.
    env.reset()
    action = {
        "home": np.array([0.0, 0.5, 1.0], dtype=np.float32),
        "away": np.array([0.0, 0.5, 1.0], dtype=np.float32),
    }
    for _ in range(max(hold_steps // 2, 10)):
        env.step(action)

    (_, _), (_, v_home) = env.sim.get_joint_positions_and_vels()
    (_, _), (_, v_away) = env.sim.get_opponent_joint_positions_and_vels()

    print(f"paddle_vel_home={v_home:+.3f}, paddle_vel_away={v_away:+.3f}")
    check(np.sign(v_home) == np.sign(v_away), "Away paddle velocity should keep same sign convention")
    print("[PASS] Away action mapping matches ShootoutVersusEnv conventions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity check ShootoutVersusEnv")
    parser.add_argument("--render", action="store_true", help="Use PyBullet GUI")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=150, help="Step count for API contract test")
    parser.add_argument("--mirror-trials", type=int, default=20)
    parser.add_argument("--hold-steps", type=int, default=40)
    parser.add_argument("--mirror-tol", type=float, default=5e-4)
    parser.add_argument("--policy-hz", type=float, default=30.0)
    parser.add_argument("--sim-hz", type=int, default=240)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = ShootoutVersusEnv(
        render_mode="human" if args.render else "none",
        seed=args.seed,
        policy_hz=args.policy_hz,
        sim_hz=args.sim_hz,
        serve_mode="random",
        real_time_gui=args.render,
    )

    try:
        test_spaces(env)
        test_reset_step_contract(env, steps=args.steps)
        test_observation_mirroring(env, trials=args.mirror_trials, tol=args.mirror_tol)
        test_action_unmirroring(env, hold_steps=args.hold_steps)

        header("ALL CHECKS PASSED")
    finally:
        env.close()


if __name__ == "__main__":
    main()
