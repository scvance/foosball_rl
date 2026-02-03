"""
Sanity check script for FoosballVersusEnv.

Tests:
1. Ball spawning and shooting directions
2. Slider action mirroring (home vs away)
3. Kicker velocity mirroring (home vs away)
4. Observation mirroring consistency
5. Visual confirmation of movements

Run:
  python sanity_check_versus.py
"""

import numpy as np
import time
import pybullet as p

from foosball_envs.FoosballVersusEnv import FoosballVersusEnv


def print_header(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_subheader(title):
    print(f"\n--- {title} ---")


def test_ball_spawning(env, num_tests=10):
    """Test that balls spawn and shoot in the correct directions."""
    print_header("TEST 1: Ball Spawning and Shooting")
    
    home_serves = 0
    away_serves = 0
    
    for i in range(num_tests):
        obs, info = env.reset()
        shot_info = info.get("shot", {})
        for _ in range(3):
            obs, _, _, _, info = env.step({"home": np.array([0.0, 0.0, 0.0], dtype=np.float32), "away": np.array([0.0, 0.0, 0.0], dtype=np.float32)})
        target = shot_info.get("target_goal", "unknown")
        
        # Get ball velocity
        ball_pos, ball_vel = env.sim.get_ball_true_local_pos_vel()
        vx = float(ball_vel[0])
        
        if target == "home":
            home_serves += 1
            expected_dir = "left (vx < 0)"
            actual_dir = "left" if vx < 0 else "right"
            correct = vx < 0
        else:
            away_serves += 1
            expected_dir = "right (vx > 0)"
            actual_dir = "right" if vx > 0 else "left"
            correct = vx > 0
        
        # ensure that the observations show mirrored ball velocity
        obs_home = obs["home"]
        obs_away = obs["away"]
        vx_home = obs_home[3]
        vy_home = obs_home[4]
        vx_away = obs_away[3]
        vy_away = obs_away[4]
        obs_correct = (abs(vx_home + vx_away) < 0.001) and (abs(vy_home + vy_away) < 0.001)
        status = "✓" if correct and obs_correct else "✗"
        print(f"  [{status}] Serve {i+1}: target={target}, vx={vx:+.2f}, expected={expected_dir}, actual={actual_dir}, home_vx={vx_home:+.2f}, away_vx={vx_away:+.2f}, vy_home={vy_home:+.2f}, vy_away={vy_away:+.2f}")
    
    print(f"\n  Summary: {home_serves} home serves, {away_serves} away serves")
    print(f"  (serve_side='random' should give ~50/50 split)")


def test_slider_mirroring(env, duration=3.0):
    """Test that slider actions are properly mirrored."""
    print_header("TEST 2: Slider Position Mirroring")
    
    obs, _ = env.reset()
    policy_hz = env.policy_hz
    steps = int(duration * policy_hz)
    
    print_subheader("Both sliders: action = +1 (should move TOWARD table center)")
    print("  Home slider should move in +Y direction")
    print("  Away slider should move in -Y direction (mirrored from +1 -> -1)")
    
    for i in range(steps):
        # Both policies output +1 for slider position
        action = {
            "home": np.array([1.0, 1.0, 0.0], dtype=np.float32),  # slider_pos=+1
            "away": np.array([1.0, 1.0, 0.0], dtype=np.float32),  # slider_pos=+1 (will be mirrored to -1)
        }
        obs, rewards, terminated, truncated, info = env.step(action)
        
        if i % 20 == 0:
            # Get actual joint positions
            slider_home, _ = env.sim.get_joint_positions()
            slider_away, _ = env.sim.get_opponent_joint_positions()
            print(f"  Step {i:3d}: home_slider={slider_home:+.4f}, away_slider={slider_away:+.4f}")
        
        # if terminated or truncated:
        #     obs, _ = env.reset()
    
    print_subheader("Both sliders: action = -1 (should move AWAY from table center)")
    obs, _ = env.reset()
    
    for i in range(steps):
        action = {
            "home": np.array([-1.0, 1.0, 0.0], dtype=np.float32),
            "away": np.array([-1.0, 1.0, 0.0], dtype=np.float32),
        }
        obs, rewards, terminated, truncated, info = env.step(action)
        
        if i % 20 == 0:
            slider_home, _ = env.sim.get_joint_positions()
            slider_away, _ = env.sim.get_opponent_joint_positions()
            print(f"  Step {i:3d}: home_slider={slider_home:+.4f}, away_slider={slider_away:+.4f}")
        
        # if terminated or truncated:
        #     obs, _ = env.reset()


def test_kicker_mirroring(env, duration=3.0):
    """Test that kicker velocity actions are properly mirrored."""
    print_header("TEST 3: Kicker Velocity Mirroring")
    
    obs, _ = env.reset()
    policy_hz = env.policy_hz
    steps = int(duration * policy_hz)
    
    print_subheader("Both kickers: action = +1 (positive angular velocity)")
    print("  Home kicker: +vel (spins one way)")
    print("  Away kicker: -vel (mirrored, spins opposite way)")
    print("  Both should appear to swing 'forward' toward opponent")
    
    for i in range(steps):
        action = {
            "home": np.array([0.0, 0.5, 1.0], dtype=np.float32),  # kicker_vel=+1
            "away": np.array([0.0, 0.5, 1.0], dtype=np.float32),  # kicker_vel=+1 (mirrored to -1)
        }
        obs, rewards, terminated, truncated, info = env.step(action)
        
        if i % 20 == 0:
            (_, kicker_home), (_, v_kicker_home) = env.sim.get_joint_positions_and_vels()
            (_, kicker_away), (_, v_kicker_away) = env.sim.get_opponent_joint_positions_and_vels()
            print(f"  Step {i:3d}: home_kicker_vel={v_kicker_home:+.1f}, away_kicker_vel={v_kicker_away:+.1f}")
        
        # if terminated or truncated:
        #     obs, _ = env.reset()
    
    print_subheader("Both kickers: action = -1 (negative angular velocity)")
    obs, _ = env.reset()
    
    for i in range(steps):
        action = {
            "home": np.array([0.0, 0.5, -1.0], dtype=np.float32),
            "away": np.array([0.0, 0.5, -1.0], dtype=np.float32),
        }
        obs, rewards, terminated, truncated, info = env.step(action)
        
        if i % 20 == 0:
            (_, kicker_home), (_, v_kicker_home) = env.sim.get_joint_positions_and_vels()
            (_, kicker_away), (_, v_kicker_away) = env.sim.get_opponent_joint_positions_and_vels()
            print(f"  Step {i:3d}: home_kicker_vel={v_kicker_home:+.1f}, away_kicker_vel={v_kicker_away:+.1f}")
        
        # if terminated or truncated:
        #     obs, _ = env.reset()
    
    print_subheader("Kicker holding: action = 0 (should hold position)")
    obs, _ = env.reset()
    
    for i in range(steps):
        action = {
            "home": np.array([0.0, 0.5, 0.0], dtype=np.float32),
            "away": np.array([0.0, 0.5, 0.0], dtype=np.float32),
        }
        obs, rewards, terminated, truncated, info = env.step(action)
        
        if i % 20 == 0:
            (_, kicker_home), (_, v_kicker_home) = env.sim.get_joint_positions_and_vels()
            (_, kicker_away), (_, v_kicker_away) = env.sim.get_opponent_joint_positions_and_vels()
            print(f"  Step {i:3d}: home_kicker_vel={v_kicker_home:+.1f}, away_kicker_vel={v_kicker_away:+.1f} (should be ~0)")
        
        # if terminated or truncated:
        #     obs, _ = env.reset()


def test_observation_mirroring(env):
    """Test that observations are properly mirrored for the away side."""
    print_header("TEST 4: Observation Mirroring")
    
    obs, info = env.reset()
    obs_home = obs["home"]
    obs_away = obs["away"]
    
    # Observation layout: [est_pos(3), est_vel(3), pred_pos(3), own_joints(4), opp_joints(4), intercept(4)]
    
    print_subheader("Ball Position (indices 0-2)")
    print(f"  Home sees ball at: [{obs_home[0]:+.4f}, {obs_home[1]:+.4f}, {obs_home[2]:+.4f}]")
    print(f"  Away sees ball at: [{obs_away[0]:+.4f}, {obs_away[1]:+.4f}, {obs_away[2]:+.4f}]")
    print(f"  (X and Y should be mirrored about table center)")
    
    # Get table center
    cx = env._table_center_x
    cy = env._table_center_y
    
    # Check if mirroring is correct
    expected_away_x = 2 * cx - obs_home[0]
    expected_away_y = 2 * cy - obs_home[1]
    x_match = abs(obs_away[0] - expected_away_x) < 0.001
    y_match = abs(obs_away[1] - expected_away_y) < 0.001
    print(f"  Table center: ({cx:.4f}, {cy:.4f})")
    print(f"  Expected away X: {expected_away_x:+.4f}, actual: {obs_away[0]:+.4f} {'✓' if x_match else '✗'}")
    print(f"  Expected away Y: {expected_away_y:+.4f}, actual: {obs_away[1]:+.4f} {'✓' if y_match else '✗'}")
    
    print_subheader("Ball Velocity (indices 3-5)")
    print(f"  Home sees velocity: [{obs_home[3]:+.4f}, {obs_home[4]:+.4f}, {obs_home[5]:+.4f}]")
    print(f"  Away sees velocity: [{obs_away[3]:+.4f}, {obs_away[4]:+.4f}, {obs_away[5]:+.4f}]")
    print(f"  (X and Y components should be negated)")
    
    vx_match = abs(obs_away[3] - (-obs_home[3])) < 0.001
    vy_match = abs(obs_away[4] - (-obs_home[4])) < 0.001
    print(f"  VX negated: {-obs_home[3]:+.4f} vs {obs_away[3]:+.4f} {'✓' if vx_match else '✗'}")
    print(f"  VY negated: {-obs_home[4]:+.4f} vs {obs_away[4]:+.4f} {'✓' if vy_match else '✗'}")
    
    print_subheader("Own Joints (indices 9-12: kicker_pos, slider_pos, kicker_vel, slider_vel)")
    print(f"  Home own joints: [{obs_home[9]:+.4f}, {obs_home[10]:+.4f}, {obs_home[11]:+.4f}, {obs_home[12]:+.4f}]")
    print(f"  Away own joints: [{obs_away[9]:+.4f}, {obs_away[10]:+.4f}, {obs_away[11]:+.4f}, {obs_away[12]:+.4f}]")
    print(f"  (Away sees its own joints, not home's)")
    
    print_subheader("Opponent Joints (indices 13-16)")
    print(f"  Home opp joints: [{obs_home[13]:+.4f}, {obs_home[14]:+.4f}, {obs_home[15]:+.4f}, {obs_home[16]:+.4f}]")
    print(f"  Away opp joints: [{obs_away[13]:+.4f}, {obs_away[14]:+.4f}, {obs_away[15]:+.4f}, {obs_away[16]:+.4f}]")
    print(f"  (Home's 'own' should match away's 'opp' and vice versa)")


def test_symmetric_policy_behavior(env, duration=5.0):
    """
    Test that identical actions produce symmetric (mirrored) behavior.
    
    If the policy outputs the same action for both sides, they should
    move symmetrically (toward/away from their respective goals).
    """
    print_header("TEST 5: Symmetric Policy Behavior")
    
    obs, _ = env.reset()
    policy_hz = env.policy_hz
    steps = int(duration * policy_hz)
    
    print_subheader("Identical actions: slider=+1, kicker=+0.5")
    print("  Both goalies should move toward the center of the table")
    print("  and spin their kickers in the 'forward' direction")
    
    for i in range(steps):
        # Same action for both
        action = np.array([0.5, 1.0, 0.5], dtype=np.float32)
        action_dict = {"home": action, "away": action.copy()}
        
        obs, rewards, terminated, truncated, info = env.step(action_dict)
        
        if i % 30 == 0:
            slider_home, kicker_home = env.sim.get_joint_positions()
            slider_away, kicker_away = env.sim.get_opponent_joint_positions()
            
            # Get player centers
            home_center = env.sim.get_player_center_local()
            away_center = env.sim.get_opponent_player_center_local()
            
            print(f"  Step {i:3d}:")
            print(f"    Home: slider={slider_home:+.3f}, center_y={home_center[1]:+.3f}")
            print(f"    Away: slider={slider_away:+.3f}, center_y={away_center[1]:+.3f}")
        
        # if terminated or truncated:
        #     print(f"  Episode ended: {info.get('event', 'unknown')}")
        #     obs, _ = env.reset()


def test_kicker_free_spin(env, duration=4.0):
    """Test that kickers can spin freely (continuous joints)."""
    print_header("TEST 6: Kicker Free Spin (Continuous Joints)")
    
    obs, _ = env.reset()
    policy_hz = env.policy_hz
    steps = int(duration * policy_hz)
    
    print("  Applying maximum spin to both kickers...")
    print("  Kickers should spin continuously without jamming")
    
    max_angle_home = 0.0
    max_angle_away = 0.0
    
    for i in range(steps):
        action = {
            "home": np.array([0.0, 0.5, 1.0], dtype=np.float32),  # Max spin
            "away": np.array([0.0, 0.5, 1.0], dtype=np.float32),
        }
        obs, rewards, terminated, truncated, info = env.step(action)
        
        _, kicker_home = env.sim.get_joint_positions()
        _, kicker_away = env.sim.get_opponent_joint_positions()
        
        max_angle_home = max(max_angle_home, abs(kicker_home))
        max_angle_away = max(max_angle_away, abs(kicker_away))
        
        if i % 20 == 0:
            (_, _), (_, v_kicker_home) = env.sim.get_joint_positions_and_vels()
            (_, _), (_, v_kicker_away) = env.sim.get_opponent_joint_positions_and_vels()
            print(f"  Step {i:3d}: home_angle={np.degrees(kicker_home):+.0f}°, away_angle={np.degrees(kicker_away):+.0f}°, "
                  f"home_vel={v_kicker_home:+.1f} rad/s, away_vel={v_kicker_away:+.1f} rad/s")
        
        # if terminated or truncated:
        #     obs, _ = env.reset()
    
    print(f"\n  Max angle reached - Home: {np.degrees(max_angle_home):.0f}°, Away: {np.degrees(max_angle_away):.0f}°")
    if max_angle_home > np.pi and max_angle_away > np.pi:
        print("  ✓ Both kickers exceeded 180° - continuous joints working!")
    else:
        print("  ✗ Kickers may be limited - check URDF joint type")


def interactive_control(env):
    """Interactive mode to manually test controls."""
    print_header("INTERACTIVE MODE")
    print("""
Controls:
  W/S - Home slider forward/backward
  A/D - Home kicker spin left/right
  
  I/K - Away slider forward/backward  
  J/L - Away kicker spin left/right
  
  SPACE - Reset episode
  Q - Quit
  
Watch the simulation window and verify movements look correct.
""")
    
    obs, _ = env.reset()
    
    home_slider = 0.0
    home_kicker = 0.0
    away_slider = 0.0
    away_kicker = 0.0
    
    print("(Interactive mode requires a GUI - press keys in the terminal)")
    print("Press Enter to step, or type a command:")
    
    try:
        while True:
            cmd = input("> ").strip().lower()
            
            if cmd == 'q':
                break
            elif cmd == 'w':
                home_slider = min(1.0, home_slider + 0.2)
            elif cmd == 's':
                home_slider = max(-1.0, home_slider - 0.2)
            elif cmd == 'a':
                home_kicker = max(-1.0, home_kicker - 0.2)
            elif cmd == 'd':
                home_kicker = min(1.0, home_kicker + 0.2)
            elif cmd == 'i':
                away_slider = min(1.0, away_slider + 0.2)
            elif cmd == 'k':
                away_slider = max(-1.0, away_slider - 0.2)
            elif cmd == 'j':
                away_kicker = max(-1.0, away_kicker - 0.2)
            elif cmd == 'l':
                away_kicker = min(1.0, away_kicker + 0.2)
            elif cmd == ' ' or cmd == 'r':
                obs, _ = env.reset()
                home_slider = home_kicker = away_slider = away_kicker = 0.0
                print("Reset!")
                continue
            
            # Step with current actions
            action = {
                "home": np.array([home_slider, 0.8, home_kicker], dtype=np.float32),
                "away": np.array([away_slider, 0.8, away_kicker], dtype=np.float32),
            }
            
            for _ in range(5):  # Step a few times to see movement
                obs, rewards, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    print(f"Episode ended: {info.get('event', 'unknown')}")
                    obs, _ = env.reset()
                    break
            
            print(f"Actions - Home: slider={home_slider:+.1f}, kicker={home_kicker:+.1f} | "
                  f"Away: slider={away_slider:+.1f}, kicker={away_kicker:+.1f}")
    
    except KeyboardInterrupt:
        print("\nExiting interactive mode.")


def run_visual_demo(env, duration=20.0):
    """Run a visual demo with random actions."""
    print_header("VISUAL DEMO")
    print(f"  Running for {duration} seconds with random actions...")
    print("  Watch the simulation to verify behavior looks correct.")
    
    obs, _ = env.reset()
    start_time = time.time()
    
    while time.time() - start_time < duration:
        action = {
            "home": env.action_space["home"].sample(),
            "away": env.action_space["away"].sample(),
        }
        obs, rewards, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            event = info.get("event", "unknown")
            print(f"  Episode ended: {event}, rewards: home={rewards['home']:.1f}, away={rewards['away']:.1f}")
            obs, _ = env.reset()
    
    print("  Demo complete!")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sanity check FoosballVersusEnv")
    parser.add_argument("--render", action="store_true", help="Enable GUI rendering")
    parser.add_argument("--interactive", action="store_true", help="Run interactive mode")
    parser.add_argument("--demo", action="store_true", help="Run visual demo only")
    parser.add_argument("--test", type=int, default=0, help="Run specific test (1-6), 0 for all")
    args = parser.parse_args()
    
    render_mode = "human" if args.render else "none"
    
    print("\n" + "=" * 70)
    print(" FOOSBALL VERSUS ENVIRONMENT SANITY CHECK")
    print("=" * 70)
    print(f"\nRender mode: {render_mode}")
    
    env = FoosballVersusEnv(
        render_mode=render_mode,
        seed=42,
        policy_hz=200.0,
        sim_hz=1000,
        serve_side="random",
        real_time_gui=True,
    )
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    try:
        if args.demo:
            run_visual_demo(env, duration=30.0)
        elif args.interactive:
            interactive_control(env)
        else:
            tests = {
                1: lambda: test_ball_spawning(env, num_tests=10),
                2: lambda: test_slider_mirroring(env, duration=2.0),
                3: lambda: test_kicker_mirroring(env, duration=2.0),
                4: lambda: test_observation_mirroring(env),
                5: lambda: test_symmetric_policy_behavior(env, duration=3.0),
                6: lambda: test_kicker_free_spin(env, duration=3.0),
            }
            
            if args.test > 0 and args.test in tests:
                tests[args.test]()
            else:
                for test_fn in tests.values():
                    test_fn()
            
            if args.render:
                input("\nPress Enter to run visual demo (or Ctrl+C to exit)...")
                run_visual_demo(env, duration=15.0)
        
        print_header("SANITY CHECK COMPLETE")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()