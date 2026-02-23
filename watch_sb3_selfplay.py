"""
Watch trained SAC agents play foosball against each other.

Loads a trained model and runs episodes with rendering enabled.
"""

import argparse
import time

import numpy as np
import torch

from stable_baselines3 import SAC

from foosball_envs.FoosballVersusEnv import FoosballVersusEnv


def main():
    parser = argparse.ArgumentParser(description="Watch trained agents play foosball")
    parser.add_argument("model_path", type=str, help="Path to trained model (e.g., best/best_model.zip or final_model.zip)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to watch")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions (no exploration)")
    parser.add_argument("--delay", type=float, default=0.001, help="Additional delay between steps (seconds)")
    
    # Env params (should match training)
    parser.add_argument("--policy_hz", type=float, default=200.0)
    parser.add_argument("--sim_hz", type=int, default=1000)
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--serve_side", type=str, default="random")
    parser.add_argument("--speed_min", type=float, default=1.0)
    parser.add_argument("--speed_max", type=float, default=15.0)
    parser.add_argument("--bounce_prob", type=float, default=0.25)
    parser.add_argument("--num_substeps", type=int, default=8)
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = SAC.load(args.model_path)
    print("Model loaded!")
    
    # Create environment with GUI rendering
    env = FoosballVersusEnv(
        render_mode="human",
        policy_hz=args.policy_hz,
        sim_hz=args.sim_hz,
        max_episode_steps=args.max_episode_steps,
        serve_side=args.serve_side,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
        bounce_prob=args.bounce_prob,
        num_substeps=args.num_substeps,
        real_time_gui=True,
        spawn_without_velocity=True,
    )
    
    # Stats tracking
    home_goals = 0
    away_goals = 0
    home_blocks = 0
    away_blocks = 0
    outs = 0
    stalls = 0
    
    print(f"\nWatching {args.episodes} episodes...")
    print("=" * 50)
    
    for ep in range(args.episodes):
        obs_dict, info = env.reset()
        done = False
        step_count = 0
        ep_reward_home = 0.0
        ep_reward_away = 0.0
        
        while not done:
            # Get observations for both sides
            home_obs = obs_dict["home"]
            away_obs = obs_dict["away"]
            
            # Get actions from policy for both sides
            home_action, _ = model.predict(home_obs, deterministic=args.deterministic)
            away_action, _ = model.predict(away_obs, deterministic=args.deterministic)
            
            # Step environment
            action_dict = {"home": home_action, "away": away_action}
            obs_dict, reward_dict, terminated, truncated, info = env.step(action_dict)
            done = terminated or truncated
            
            ep_reward_home += reward_dict["home"]
            ep_reward_away += reward_dict["away"]
            step_count += 1
            
            if args.delay > 0:
                time.sleep(args.delay)
        
        # Track stats
        event = info.get("event", "unknown")
        if event == "home_goal":
            home_goals += 1
            result = "HOME GOAL!"
        elif event == "away_goal":
            away_goals += 1
            result = "AWAY GOAL!"
        elif event == "home_block":
            home_blocks += 1
            result = "Home block"
        elif event == "away_block":
            away_blocks += 1
            result = "Away block"
        elif event == "out":
            outs += 1
            result = "Ball out"
        elif event == "stalled":
            stalls += 1
            result = "Stalled"
        else:
            result = event
        
        print(f"Episode {ep + 1:3d}: {result:12s} | Steps: {step_count:4d} | "
              f"Rewards: home={ep_reward_home:7.2f}, away={ep_reward_away:7.2f}")
    
    env.close()
    
    # Print summary
    print("=" * 50)
    print("\nSummary:")
    print(f"  Home goals:  {home_goals}")
    print(f"  Away goals:  {away_goals}")
    print(f"  Home blocks: {home_blocks}")
    print(f"  Away blocks: {away_blocks}")
    print(f"  Outs:        {outs}")
    print(f"  Stalls:      {stalls}")
    
    total_episodes = args.episodes
    print(f"\nRates (over {total_episodes} episodes):")
    print(f"  Goal rate:  {(home_goals + away_goals) / total_episodes:.1%}")
    print(f"  Block rate: {(home_blocks + away_blocks) / total_episodes:.1%}")


if __name__ == "__main__":
    main()