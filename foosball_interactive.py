#!/usr/bin/env python3
"""
Interactive terminal controller for FoosballVersusEnv.

Commands:
  home <slider_pos> <slider_vel> <kicker_vel>   - Set home action (values in [-1, 1])
  away <slider_pos> <slider_vel> <kicker_vel>   - Set away action (values in [-1, 1])
  both <slider_pos> <slider_vel> <kicker_vel>   - Set both to same action
  step                                           - Step sim 1000 times (1 second) and show obs
  reset                                          - Reset the environment
  obs                                            - Show current observation without stepping
  quit / q                                       - Exit

Example:
  home 0.5 1.0 0.0    # Move home slider right, full speed, no kicker spin
  away -0.5 0.5 1.0   # Move away slider left, half speed, full kicker spin
  step                # Execute for 1 second and see results
"""

import numpy as np
import sys
import os

# Add parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from FoosballVersusEnv import FoosballVersusEnv


def format_obs(obs: dict, side: str) -> str:
    """Format observation array into readable string."""
    o = obs[side]
    
    lines = [
        f"=== {side.upper()} Observation (shape: {o.shape}) ===",
        f"  Ball Est Pos (x,y,z):   [{o[0]:+.4f}, {o[1]:+.4f}, {o[2]:+.4f}]",
        f"  Ball Est Vel (x,y,z):   [{o[3]:+.4f}, {o[4]:+.4f}, {o[5]:+.4f}]",
        f"  Ball Pred Pos (x,y,z):  [{o[6]:+.4f}, {o[7]:+.4f}, {o[8]:+.4f}]",
        f"  Own Joints (kicker, slider, v_kicker, v_slider):",
        f"                          [{o[9]:+.4f}, {o[10]:+.4f}, {o[11]:+.4f}, {o[12]:+.4f}]",
        f"  Opp Joints (kicker, slider, v_kicker, v_slider):",
        f"                          [{o[13]:+.4f}, {o[14]:+.4f}, {o[15]:+.4f}, {o[16]:+.4f}]",
        f"  Intercept (y_pred, z_pred, x_goal, t_goal):",
        f"                          [{o[17]:+.4f}, {o[18]:+.4f}, {o[19]:+.4f}, {o[20]:+.4f}]",
    ]
    return "\n".join(lines)


def format_rewards(rewards: dict) -> str:
    return f"Rewards: home={rewards['home']:+.2f}, away={rewards['away']:+.2f}"


def main():
    print("Initializing FoosballVersusEnv with GUI...")
    env = FoosballVersusEnv(
        render_mode="human",
        seed=42,
        policy_hz=200.0,
        sim_hz=1000,
        max_episode_steps=10000,  # Long episodes for manual control
        serve_side="random",
        real_time_gui=False,  # Don't slow down for real-time in manual mode
    )
    
    obs, info = env.reset()
    print("\nEnvironment reset!")
    print(f"Shot info: {info.get('shot', 'N/A')}")
    
    # Current actions (start at neutral)
    home_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    away_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    print("\n" + "="*60)
    print("FOOSBALL INTERACTIVE CONTROLLER")
    print("="*60)
    print(__doc__)
    print(f"\nCurrent actions:")
    print(f"  Home: {home_action}")
    print(f"  Away: {away_action}")
    print()
    
    while True:
        try:
            cmd = input(">>> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        
        if not cmd:
            continue
        
        parts = cmd.split()
        command = parts[0]
        
        if command in ("quit", "q", "exit"):
            print("Exiting...")
            break
        
        elif command == "reset":
            obs, info = env.reset()
            home_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            away_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            print("\nEnvironment reset!")
            print(f"Shot info: {info.get('shot', 'N/A')}")
            print(f"Actions reset to neutral.")
        
        elif command == "obs":
            print("\n" + format_obs(obs, "home"))
            print("\n" + format_obs(obs, "away"))
            print()
        
        elif command == "home" and len(parts) == 4:
            try:
                home_action = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32)
                home_action = np.clip(home_action, -1.0, 1.0)
                print(f"Home action set to: {home_action}")
            except ValueError:
                print("Error: Values must be numbers in [-1, 1]")
        
        elif command == "away" and len(parts) == 4:
            try:
                away_action = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32)
                away_action = np.clip(away_action, -1.0, 1.0)
                print(f"Away action set to: {away_action}")
            except ValueError:
                print("Error: Values must be numbers in [-1, 1]")
        
        elif command == "both" and len(parts) == 4:
            try:
                action = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32)
                action = np.clip(action, -1.0, 1.0)
                home_action = action.copy()
                away_action = action.copy()
                print(f"Both actions set to: {action}")
            except ValueError:
                print("Error: Values must be numbers in [-1, 1]")
        
        elif command == "step":
            print(f"\nStepping with actions:")
            print(f"  Home: {home_action}")
            print(f"  Away: {away_action}")
            print("Running 1000 sim steps (1 second)...")
            
            # The env steps_per_policy is based on policy_hz
            # At policy_hz=20, each step() call does 50 sim steps
            # To get 1000 sim steps, we need 1000/50 = 20 policy steps
            steps_per_policy = env.steps_per_policy
            num_policy_steps = 1000 // steps_per_policy
            
            total_rewards = {"home": 0.0, "away": 0.0}
            terminated = False
            truncated = False
            
            for i in range(10):
                action = {"home": home_action, "away": away_action}
                obs, rewards, terminated, truncated, info = env.step(action)
                total_rewards["home"] += rewards["home"]
                total_rewards["away"] += rewards["away"]
                
                if terminated or truncated:
                    print(f"\n*** Episode ended at step {i+1}: {info.get('event', 'unknown')} ***")
                    break
            
            print("\n" + "="*60)
            print("OBSERVATION AFTER 1 SECOND")
            print("="*60)
            print("\n" + format_obs(obs, "home"))
            print("\n" + format_obs(obs, "away"))
            print(f"\n{format_rewards(total_rewards)}")
            print(f"Event: {info.get('event', 'None')}")
            
            if terminated or truncated:
                print("\nEpisode ended. Use 'reset' to start a new episode.")
            print()
        
        else:
            print("Unknown command or wrong number of arguments.")
            print("Try: home/away/both <slider_pos> <slider_vel> <kicker_vel>")
            print("     step, reset, obs, quit")
    
    env.sim.disconnect()
    print("Disconnected from simulation.")


if __name__ == "__main__":
    main()