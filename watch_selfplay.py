"""
Visualize two trained SAC agents playing in FoosballVersusEnv.
Usage:
  python3 watch_selfplay.py --home_model path/to/home.zip --away_model path/to/away.zip --episodes 5
"""

import argparse

from stable_baselines3 import SAC

from FoosballVersusEnv import FoosballVersusEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--home_model", type=str, required=True)
    parser.add_argument("--away_model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--serve_side", type=str, default="random", choices=["home", "away", "random"])
    parser.add_argument("--policy_hz", type=float, default=200.0, help="Policy control rate (Hz).")
    parser.add_argument("--sim_hz", type=int, default=1000, help="Simulation rate (Hz).")
    parser.add_argument("--slider_vel_cap_mps", type=float, default=20.0)
    parser.add_argument("--kicker_vel_cap_rads", type=float, default=170.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = FoosballVersusEnv(
        render_mode="human",
        serve_side=args.serve_side,
        policy_hz=args.policy_hz,
        sim_hz=args.sim_hz,
        slider_vel_cap_mps=args.slider_vel_cap_mps,
        kicker_vel_cap_rads=args.kicker_vel_cap_rads,
        seed=args.seed,
    )
    model_home = SAC.load(args.home_model, device="auto")
    model_away = SAC.load(args.away_model, device="auto")

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        ep_steps = 0
        while not done:
            action_home, _ = model_home.predict(obs["home"], deterministic=True)
            action_away, _ = model_away.predict(obs["away"], deterministic=True)
            obs, rewards, terminated, truncated, info = env.step({"home": action_home, "away": action_away})
            done = terminated or truncated
            ep_steps += 1
        print(f"Episode {ep} finished in {ep_steps} steps with event={info.get('event')} rewards={rewards}")
    env.close()


if __name__ == "__main__":
    main()
