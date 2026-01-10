"""
Load and watch the best SAC model trained on FoosballGoalieEnv.

Examples:
  python watch_best_sac_foosball.py --run_dir ./sac_foosball_20260110_123456
  python watch_best_sac_foosball.py --model ./sac_foosball_20260110_123456/best/best_model.zip

Optional video:
  python watch_best_sac_foosball.py --run_dir ./sac_foosball_... --record_mp4

Notes:
- This uses render_mode="human" (PyBullet GUI).
- It prints episode outcomes (goal/block/out) using info["event"] from your env.
"""

import os
import time
import argparse

import numpy as np
import pybullet as p
from stable_baselines3 import SAC

from FoosballGoalieEnv import FoosballGoalieEnv


def resolve_model_path(run_dir: str | None, model_path: str | None) -> str:
    if model_path is not None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"--model path does not exist: {model_path}")
        return model_path

    if run_dir is None:
        raise ValueError("Provide either --run_dir or --model")

    # Default SB3 EvalCallback best model location:
    # <run_dir>/best/best_model.zip
    candidate = os.path.join(run_dir, "best", "best_model.zip")
    if not os.path.exists(candidate):
        # fall back to final_model.zip (if you want)
        fallback = os.path.join(run_dir, "final_model.zip")
        if os.path.exists(fallback):
            return fallback
        raise FileNotFoundError(
            f"Could not find best model at: {candidate}\n"
            f"and no final model at: {fallback}"
        )
    return candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default=None, help="Training run directory (contains best/best_model.zip).")
    parser.add_argument("--model", type=str, default=None, help="Explicit path to model .zip (overrides --run_dir).")

    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to watch.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions.")
    parser.add_argument("--seed", type=int, default=0)

    # Env params (should match training unless you intentionally domain-randomize)
    parser.add_argument("--speed_min", type=float, default=2.0)
    parser.add_argument("--speed_max", type=float, default=10.0)
    parser.add_argument("--bounce_prob", type=float, default=0.25)
    parser.add_argument("--action_repeat", type=int, default=8)
    parser.add_argument("--max_episode_steps", type=int, default=1500)
    parser.add_argument("--time_step", type=float, default=1.0 / 240.0)

    # Optional recording
    parser.add_argument("--record_mp4", action="store_true", help="Record MP4 using PyBullet state logging.")
    parser.add_argument("--video_dir", type=str, default=None, help="Where to write MP4 (defaults to run_dir/videos).")

    args = parser.parse_args()

    model_path = resolve_model_path(args.run_dir, args.model)
    print(f"Loading model: {model_path}")

    # Create GUI env
    env = FoosballGoalieEnv(
        render_mode="human",
        seed=args.seed,
        time_step=args.time_step,
        action_repeat=args.action_repeat,
        max_episode_steps=args.max_episode_steps,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
        bounce_prob=args.bounce_prob,
    )

    # Load model
    model = SAC.load(model_path, device="auto")

    # Optional video recording setup (one file per run)
    log_id = None
    if args.record_mp4:
        if args.video_dir is None:
            if args.run_dir is None:
                args.video_dir = "./videos"
            else:
                args.video_dir = os.path.join(args.run_dir, "videos")
        os.makedirs(args.video_dir, exist_ok=True)
        mp4_path = os.path.join(args.video_dir, f"watch_{int(time.time())}.mp4")
        print(f"Recording MP4 to: {mp4_path}")
        log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, mp4_path)

    try:
        total_goals = 0
        total_blocks = 0
        total_outs = 0

        for ep in range(1, args.episodes + 1):
            obs, info = env.reset(seed=args.seed + ep)
            done = False
            ep_rew = 0.0
            ep_len = 0
            final_event = None
            out_reason = None

            while not done:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                ep_rew += float(reward)
                ep_len += 1

                # keep latest terminal metadata
                if info.get("event", None) is not None:
                    final_event = info.get("event")
                if "out_reason" in info:
                    out_reason = info["out_reason"]

                # slow down a touch if you want more watchable speed
                # time.sleep(0.01)

            if final_event == "goal":
                total_goals += 1
            elif final_event == "block":
                total_blocks += 1
            elif final_event == "out":
                total_outs += 1

            extra = ""
            if final_event == "out" and out_reason:
                extra = f" ({out_reason})"

            print(
                f"[EP {ep:03d}] reward={ep_rew:8.3f} len={ep_len:4d} "
                f"event={final_event}{extra} "
                f"(goals={total_goals}, blocks={total_blocks}, outs={total_outs})"
            )

        print("\nSummary:")
        print(f"  episodes: {args.episodes}")
        print(f"  goals:    {total_goals}")
        print(f"  blocks:   {total_blocks}")
        print(f"  outs:     {total_outs}")

    finally:
        if log_id is not None:
            try:
                p.stopStateLogging(log_id)
            except Exception:
                pass
        env.close()


if __name__ == "__main__":
    main()
