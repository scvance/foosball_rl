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
    parser.add_argument("--action_repeat", type=int, default=2)
    parser.add_argument("--serve_side", type=str, default="random", choices=["home", "away", "random"])
    args = parser.parse_args()

    env = FoosballVersusEnv(render_mode="human", action_repeat=args.action_repeat, serve_side=args.serve_side)
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
