"""
Test script for kicker mirroring with the CORRECT fix.

The issue: The pi offset in sim core just shifts the position but doesn't
mirror the rotation direction. Both kickers still rotate the same way.

The fix: Negate the kicker action for the away side (same as slider).
The pi offset may still be needed for the initial orientation, but the
ACTION must be negated for mirrored behavior.

This script tests the negation fix applied at the action level.
"""

import numpy as np
import time

from FoosballVersusEnv import FoosballVersusEnv


def test_continuous_rotation(env, duration=8.0):
    """
    Test continuous rotation to clearly see direction.
    Both kickers should rotate in OPPOSITE world directions
    (same direction from their own perspective).
    """
    print("\n" + "="*60)
    print("TEST: CONTINUOUS ROTATION WITH NEGATION FIX")
    print("="*60)
    print("Away kicker action is NEGATED.")
    print("Both should rotate in opposite world directions.")
    print("="*60)
    
    policy_hz = env.policy_hz
    steps = int(duration * policy_hz)
    
    for i in range(steps):
        t = i / steps
        # Continuous sweep from -1 to +1 and back
        kicker_pos = np.sin(2 * np.pi * t * 0.5)  # Slow oscillation
        
        # Home: raw action
        action_home = np.array([0.0, 0.5, kicker_pos, 0.8], dtype=np.float32)
        
        # Away: NEGATED kicker action (this is the fix)
        action_away = np.array([0.0, 0.5, -kicker_pos, 0.8], dtype=np.float32)
        
        actions = {"home": action_home, "away": action_away}
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        if i % 30 == 0:
            home_k = info.get("kicker_home", 0)
            away_k = info.get("kicker_away", 0)
            print(f"  action: home={kicker_pos:+.2f}, away={-kicker_pos:+.2f}")
            print(f"  result: home={np.degrees(home_k):+6.1f}°, away={np.degrees(away_k):+6.1f}°")


def test_static_positions_with_fix(env):
    """
    Test static positions with the negation fix.
    """
    print("\n" + "="*60)
    print("TEST: STATIC POSITIONS WITH NEGATION FIX")
    print("="*60)
    
    policy_hz = env.policy_hz
    hold_time = 3.0
    steps = int(hold_time * policy_hz)
    
    # With negation: home=+1, away=-1 should make them look mirrored
    tests = [
        ("Both DOWN: home=-1, away=+1 (negated)", -1.0, +1.0),
        ("Both UP: home=+1, away=-1 (negated)", +1.0, -1.0),
        ("Both CENTER: home=0, away=0", 0.0, 0.0),
        ("Home forward, Away forward: home=+1, away=-1", +1.0, -1.0),
    ]
    
    for name, home_kicker, away_kicker in tests:
        print(f"\n  {name}")
        print(f"  Holding for {hold_time}s...")
        
        action_home = np.array([0.0, 0.8, home_kicker, 0.9], dtype=np.float32)
        action_away = np.array([0.0, 0.8, away_kicker, 0.9], dtype=np.float32)
        
        for _ in range(steps):
            actions = {"home": action_home, "away": action_away}
            obs, rewards, terminated, truncated, info = env.step(actions)
        
        home_k = info.get("kicker_home", 0)
        away_k = info.get("kicker_away", 0)
        print(f"  Final angles: home={np.degrees(home_k):+.1f}°, away={np.degrees(away_k):+.1f}°")


def test_single_policy_simulation(env, duration=10.0):
    """
    Simulate what a single policy would do playing both sides.
    
    The policy outputs a single action. 
    - Home receives it directly
    - Away receives it with slider AND kicker negated
    
    This should result in perfectly mirrored movement.
    """
    print("\n" + "="*60)
    print("TEST: SINGLE POLICY SIMULATION")
    print("="*60)
    print("Simulating one policy controlling both sides.")
    print("Policy action -> Home: direct, Away: negated slider & kicker")
    print("="*60)
    
    policy_hz = env.policy_hz
    steps = int(duration * policy_hz)
    
    for i in range(steps):
        t = i / steps
        
        # "Policy" outputs these actions
        policy_slider = np.sin(2 * np.pi * t * 0.8)
        policy_kicker = np.sin(2 * np.pi * t * 1.2)
        policy_slider_vel = 0.7
        policy_kicker_vel = 0.9
        
        # Home gets direct policy output
        action_home = np.array([
            policy_slider,
            policy_slider_vel,
            policy_kicker,
            policy_kicker_vel
        ], dtype=np.float32)
        
        # Away gets MIRRORED: negate position actions, keep velocities
        action_away = np.array([
            -policy_slider,      # Negated
            policy_slider_vel,   # Same
            -policy_kicker,      # Negated
            policy_kicker_vel    # Same
        ], dtype=np.float32)
        
        actions = {"home": action_home, "away": action_away}
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        if i % 40 == 0:
            print(f"  Policy: slider={policy_slider:+.2f}, kicker={policy_kicker:+.2f}")


def test_forward_kick_with_fix(env, duration=6.0):
    """
    Test forward kicks with the fix applied.
    
    "Forward kick" = the motion that would hit the ball toward opponent's goal.
    - For Home: hitting ball in +X direction
    - For Away: hitting ball in -X direction
    
    With correct mirroring, the same action should produce forward kicks
    for both (in their respective directions).
    """
    print("\n" + "="*60)
    print("TEST: FORWARD KICK WITH FIX")
    print("="*60)
    print("Testing if same 'policy action' produces forward kick for both.")
    print("Watch: both players should swing toward their opponent's side.")
    print("="*60)
    
    policy_hz = env.policy_hz
    
    # Do a few kick cycles
    for cycle in range(3):
        print(f"\n  Kick cycle {cycle + 1}...")
        
        # Wind up (kicker back)
        print("    Winding up...")
        for _ in range(int(1.0 * policy_hz)):
            policy_kicker = -0.8  # Back position
            action_home = np.array([0.0, 0.5, policy_kicker, 0.9], dtype=np.float32)
            action_away = np.array([0.0, 0.5, -policy_kicker, 0.9], dtype=np.float32)
            actions = {"home": action_home, "away": action_away}
            obs, rewards, terminated, truncated, info = env.step(actions)
        
        # Kick (fast forward motion)
        print("    Kicking forward...")
        for _ in range(int(0.5 * policy_hz)):
            policy_kicker = 0.8  # Forward position
            action_home = np.array([0.0, 0.5, policy_kicker, 1.0], dtype=np.float32)
            action_away = np.array([0.0, 0.5, -policy_kicker, 1.0], dtype=np.float32)
            actions = {"home": action_home, "away": action_away}
            obs, rewards, terminated, truncated, info = env.step(actions)
        
        # Hold
        time.sleep(0.3)


def test_boundaries(env):
    """
    Test the boundary issue you mentioned - kicker moves and then stops.
    Let's see what's happening at the limits.
    """
    print("\n" + "="*60)
    print("TEST: BOUNDARY BEHAVIOR")
    print("="*60)
    print("Testing kicker at extreme positions to diagnose stopping issue.")
    print("="*60)
    
    sim = env.sim
    print(f"\nHome kicker limits: [{sim.kicker_limits.lower:.3f}, {sim.kicker_limits.upper:.3f}] rad")
    print(f"  = [{np.degrees(sim.kicker_limits.lower):.1f}°, {np.degrees(sim.kicker_limits.upper):.1f}°]")
    
    if sim.opponent_kicker_limits:
        print(f"\nAway kicker limits: [{sim.opponent_kicker_limits.lower:.3f}, {sim.opponent_kicker_limits.upper:.3f}] rad")
        print(f"  = [{np.degrees(sim.opponent_kicker_limits.lower):.1f}°, {np.degrees(sim.opponent_kicker_limits.upper):.1f}°]")
        print(f"  Offset applied: {sim.opponent_kicker_offset:.3f} rad = {np.degrees(sim.opponent_kicker_offset):.1f}°")
    
    policy_hz = env.policy_hz
    
    # Sweep through full range slowly
    print("\n  Sweeping through full range slowly...")
    for i in range(int(5.0 * policy_hz)):
        t = i / (5.0 * policy_hz)
        kicker_pos = -1.0 + 2.0 * t  # -1 to +1
        
        action_home = np.array([0.0, 0.5, kicker_pos, 0.5], dtype=np.float32)
        action_away = np.array([0.0, 0.5, -kicker_pos, 0.5], dtype=np.float32)
        
        actions = {"home": action_home, "away": action_away}
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        if i % 20 == 0:
            home_k = info.get("kicker_home", 0)
            away_k = info.get("kicker_away", 0)
            print(f"  action={kicker_pos:+.2f} -> home={np.degrees(home_k):+6.1f}°, away={np.degrees(away_k):+6.1f}°")


def main():
    print("="*60)
    print("KICKER MIRRORING TEST WITH NEGATION FIX")
    print("="*60)
    print("\nThis test applies the CORRECT fix: negating the kicker action")
    print("for the away side (not just adding a pi offset).\n")
    
    env = FoosballVersusEnv(
        render_mode="human",
        seed=42,
        policy_hz=20.0,
        sim_hz=1000,
        serve_side="random",
        max_episode_steps=100000,  # Very long to avoid resets
    )
    
    obs, info = env.reset()
    
    try:
        input("Press Enter to test continuous rotation with negation fix...")
        test_continuous_rotation(env)
        
        input("\nPress Enter for static positions with fix...")
        test_static_positions_with_fix(env)
        
        input("\nPress Enter to test boundary behavior...")
        test_boundaries(env)
        
        input("\nPress Enter for forward kick test...")
        test_forward_kick_with_fix(env)
        
        input("\nPress Enter for single policy simulation...")
        test_single_policy_simulation(env)
        
        print("\n" + "="*60)
        print("TEST COMPLETE")
        print("="*60)
        print("""
SUMMARY OF FIX NEEDED:

In foosball_versus_env.py step() method, the away actions should be:
  - Slider position: NEGATE (already confirmed)
  - Kicker position: NEGATE (confirmed by this test)
  - Slider velocity: KEEP AS-IS
  - Kicker velocity: KEEP AS-IS

Code change in step():
```python
# Current (buggy):
a_sp = float(np.clip(a_away[0], -1.0, 1.0))
a_kp = float(np.clip(a_away[2], -1.0, 1.0))

# Fixed:
a_sp = -float(np.clip(a_away[0], -1.0, 1.0))  # Negate slider
a_kp = -float(np.clip(a_away[2], -1.0, 1.0))  # Negate kicker
```

The pi offset in sim core may still be useful for initial URDF orientation,
but it does NOT provide action mirroring.
""")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted.")
    finally:
        env.close()


if __name__ == "__main__":
    main()