"""
watch_shootout_selfplay.py

Load a trained SAC model for ShootoutVersusEnv and watch it play against itself.

Example:
  python train_scripts/watch_shootout_selfplay.py \
    --model_path runs/sac_shootout_YYYYMMDD_HHMMSS/final_model.zip
"""

import argparse
import time

from stable_baselines3 import SAC

from foosball_envs.ShootoutVersusEnv import ShootoutVersusEnv
from train_scripts.train_shootout_versus import ScriptedPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to SAC model .zip")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions")
    parser.add_argument(
        "--opponent",
        type=str,
        default=None,
        help="Away opponent: 'scripted' for rule-based, path to a .zip for a different model, "
             "or omit to use the same policy for both sides (self-play).",
    )

    # Keep these aligned with training defaults unless you intentionally change them.
    parser.add_argument("--policy_hz", type=float, default=200.0)
    parser.add_argument("--sim_hz", type=int, default=1000)
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument(
        "--serve_mode",
        type=str,
        default="random",
        choices=["random_fire", "corner", "random"],
    )
    parser.add_argument("--handle_vel_cap_mps", type=float, default=10.0)
    parser.add_argument("--paddle_vel_cap_rads", type=float, default=20.0)
    parser.add_argument("--ball_restitution", type=float, default=0.30)
    parser.add_argument("--wall_restitution", type=float, default=0.85)
    parser.add_argument("--paddle_restitution", type=float, default=0.85)
    parser.add_argument("--num_substeps", type=int, default=1)

    return parser.parse_args()


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

    model = SAC.load(args.model_path, device="auto")

    if args.opponent is None:
        away_model = model
        print("Opponent: self (same policy both sides)")
    elif args.opponent == "scripted":
        away_model = ScriptedPolicy(n_envs=1)
        print("Opponent: scripted rule-based policy")
    else:
        away_model = SAC.load(args.opponent, device="auto")
        print(f"Opponent: {args.opponent}")

    try:
        for ep in range(1, args.episodes + 1):
            obs, info = env.reset()
            done = False
            ep_home_reward = 0.0
            ep_away_reward = 0.0
            steps = 0

            if hasattr(away_model, "on_episode_start"):
                away_model.on_episode_start(0)

            while not done:
                home_action, _ = model.predict(obs["home"], deterministic=args.deterministic)
                away_action, _ = away_model.predict(obs["away"], deterministic=args.deterministic)

                obs, rewards, terminated, truncated, info = env.step(
                    {"home": home_action, "away": away_action}
                )
                ep_home_reward += float(rewards["home"])
                ep_away_reward += float(rewards["away"])
                steps += 1
                done = bool(terminated or truncated)

            print(
                f"Episode {ep}/{args.episodes} | steps={steps} | "
                f"event={info.get('event')} | "
                f"home_reward={ep_home_reward:.3f} | away_reward={ep_away_reward:.3f}"
            )

            # Small pause so episode transitions are easier to see.
            time.sleep(0.3)
    finally:
        env.close()


if __name__ == "__main__":
    main()
