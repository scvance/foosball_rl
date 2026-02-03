"""
Test script to verify both goalies have symmetric controls.

Expected behavior:
- slider = 0 → player at Y = 0 (centered)
- slider = +0.154 → player at Y = -0.154 (one side)
- slider = -0.154 → player at Y = +0.154 (other side)
- kicker = 0 → toes pointing down
- kicker = +π/2 → toes pointing one way
- kicker = -π/2 → toes pointing the other way
"""

import numpy as np
import pybullet as p
import time

from foosball_envs.FoosballSimCore import _FoosballSimCore


def main():
    sim = _FoosballSimCore(use_gui=True, time_step=1/240, seed=42)
    
    # Get joint indices
    home_slider_idx = sim.slider_idx
    home_kicker_idx = sim.kicker_idx
    away_slider_idx = sim.opponent_slider_idx
    away_kicker_idx = sim.opponent_kicker_idx
    
    print("Joint indices:")
    print(f"  Home slider: {home_slider_idx}, kicker: {home_kicker_idx}")
    print(f"  Away slider: {away_slider_idx}, kicker: {away_kicker_idx}")
    
    print("\nJoint limits:")
    print(f"  Home slider: [{sim.slider_limits.lower:.4f}, {sim.slider_limits.upper:.4f}]")
    print(f"  Away slider: [{sim.opponent_slider_limits.lower:.4f}, {sim.opponent_slider_limits.upper:.4f}]")
    
    # Test positions
    slider_positions = [0.0, 0.154, -0.154, 0.0]
    kicker_positions = [0.0, np.pi/2, -np.pi/2, np.pi, 0.0]
    
    print("\n" + "="*70)
    print(" TEST 1: SLIDER CENTERING (slider = 0 should give player Y = 0)")
    print("="*70)
    
    # Reset both to 0
    p.resetJointState(sim.robot_uid, home_slider_idx, 0.0)
    p.resetJointState(sim.robot_uid, away_slider_idx, 0.0)
    p.resetJointState(sim.robot_uid, home_kicker_idx, 0.0)
    p.resetJointState(sim.robot_uid, away_kicker_idx, 0.0)
    
    for _ in range(10):
        p.stepSimulation()
    
    home_center = sim.get_player_center_local()
    away_center = sim.get_opponent_player_center_local()
    
    print(f"\nSlider = 0.0:")
    print(f"  Home player Y: {home_center[1]:+.4f} (expected: 0.0)")
    print(f"  Away player Y: {away_center[1]:+.4f} (expected: 0.0)")
    
    home_ok = abs(home_center[1]) < 0.01
    away_ok = abs(away_center[1]) < 0.01
    print(f"  Home centered: {'✓' if home_ok else '✗'}")
    print(f"  Away centered: {'✓' if away_ok else '✗'}")
    
    input("\nPress Enter to test slider range...")
    
    print("\n" + "="*70)
    print(" TEST 2: SLIDER RANGE")
    print("="*70)
    
    for slider_val in slider_positions:
        p.resetJointState(sim.robot_uid, home_slider_idx, slider_val)
        p.resetJointState(sim.robot_uid, away_slider_idx, slider_val)
        
        for _ in range(10):
            p.stepSimulation()
        
        home_center = sim.get_player_center_local()
        away_center = sim.get_opponent_player_center_local()
        
        print(f"\nSlider = {slider_val:+.3f}:")
        print(f"  Home player Y: {home_center[1]:+.4f}")
        print(f"  Away player Y: {away_center[1]:+.4f}")
        
        # Check symmetry: both should have same Y when given same slider value
        y_match = abs(home_center[1] - away_center[1]) < 0.01
        print(f"  Y values match: {'✓' if y_match else '✗'}")
        
        time.sleep(10.0)
    
    input("\nPress Enter to test kicker rotation...")
    
    print("\n" + "="*70)
    print(" TEST 3: KICKER ROTATION (0 = toes down)")
    print("="*70)
    
    # Reset sliders to center
    p.resetJointState(sim.robot_uid, home_slider_idx, 0.0)
    p.resetJointState(sim.robot_uid, away_slider_idx, 0.0)
    
    for kicker_val in kicker_positions:
        p.resetJointState(sim.robot_uid, home_kicker_idx, kicker_val)
        p.resetJointState(sim.robot_uid, away_kicker_idx, kicker_val)
        
        for _ in range(10):
            p.stepSimulation()
        
        degrees = np.degrees(kicker_val)
        print(f"\nKicker = {kicker_val:+.3f} rad ({degrees:+.0f}°)")
        print(f"  Look at the GUI - both kickers should be oriented the same way")
        print(f"  0° = toes down, +90° = toes toward opponent, -90° = toes away")
        
        time.sleep(1.5)
    
    input("\nPress Enter to test mirrored actions (what the policy sees)...")
    
    print("\n" + "="*70)
    print(" TEST 4: MIRRORED ACTIONS FOR SELF-PLAY")
    print("="*70)
    print("""
    For self-play, when the policy outputs action = +1 for slider:
    - Home: slider moves to +0.154 (player Y = -0.154)
    - Away: action is mirrored to -1, slider moves to -0.154 (player Y = +0.154)
    
    Both players should move toward the SAME side of the table (from their perspective).
    """)
    
    # Simulate what happens when both policies output the same action
    print("Both policies output slider_action = +1.0:")
    
    # Home: action +1 -> slider = +0.154
    home_slider = 0.154
    # Away: action +1 is mirrored to -1 -> slider = -0.154
    away_slider = -0.154
    
    p.resetJointState(sim.robot_uid, home_slider_idx, home_slider)
    p.resetJointState(sim.robot_uid, away_slider_idx, away_slider)
    
    for _ in range(10):
        p.stepSimulation()
    
    home_center = sim.get_player_center_local()
    away_center = sim.get_opponent_player_center_local()
    
    print(f"  Home: slider={home_slider:+.3f} → player Y={home_center[1]:+.4f}")
    print(f"  Away: slider={away_slider:+.3f} → player Y={away_center[1]:+.4f}")
    print(f"  (Players should be on OPPOSITE sides of Y=0, symmetric about center)")
    
    symmetric = abs(home_center[1] + away_center[1]) < 0.01
    print(f"  Symmetric: {'✓' if symmetric else '✗'}")
    
    time.sleep(2.0)
    
    print("\nBoth policies output slider_action = -1.0:")
    
    home_slider = -0.154
    away_slider = 0.154  # mirrored
    
    p.resetJointState(sim.robot_uid, home_slider_idx, home_slider)
    p.resetJointState(sim.robot_uid, away_slider_idx, away_slider)
    
    for _ in range(10):
        p.stepSimulation()
    
    home_center = sim.get_player_center_local()
    away_center = sim.get_opponent_player_center_local()
    
    print(f"  Home: slider={home_slider:+.3f} → player Y={home_center[1]:+.4f}")
    print(f"  Away: slider={away_slider:+.3f} → player Y={away_center[1]:+.4f}")
    
    symmetric = abs(home_center[1] + away_center[1]) < 0.01
    print(f"  Symmetric: {'✓' if symmetric else '✗'}")
    
    time.sleep(2.0)
    
    input("\nPress Enter to test kicker mirroring...")
    
    print("\n" + "="*70)
    print(" TEST 5: MIRRORED KICKER ACTIONS")
    print("="*70)
    print("""
    For self-play, when the policy outputs kicker_vel = +1:
    - Home: spins in + direction
    - Away: action is mirrored to -1, spins in - direction
    
    Both should appear to kick "forward" toward their opponent.
    """)
    
    # Reset to center
    p.resetJointState(sim.robot_uid, home_slider_idx, 0.0)
    p.resetJointState(sim.robot_uid, away_slider_idx, 0.0)
    
    print("Both policies output kicker_action = +1.0 (watch the spin direction):")
    
    # Home: +1 -> positive angular velocity
    # Away: +1 mirrored to -1 -> negative angular velocity
    home_kicker_target = np.pi/2
    away_kicker_target = -np.pi/2  # mirrored
    
    p.resetJointState(sim.robot_uid, home_kicker_idx, home_kicker_target)
    p.resetJointState(sim.robot_uid, away_kicker_idx, away_kicker_target)
    
    for _ in range(10):
        p.stepSimulation()
    
    print(f"  Home kicker: {np.degrees(home_kicker_target):+.0f}°")
    print(f"  Away kicker: {np.degrees(away_kicker_target):+.0f}°")
    print("  Both should appear to be rotated 'forward' toward their opponent's goal")
    
    time.sleep(2.0)
    
    print("\n" + "="*70)
    print(" TEST 6: CONTINUOUS SPIN TEST")
    print("="*70)
    
    # Reset kickers
    p.resetJointState(sim.robot_uid, home_kicker_idx, 0.0)
    p.resetJointState(sim.robot_uid, away_kicker_idx, 0.0)
    
    print("Spinning both kickers continuously (same policy action = +1)...")
    print("Home spins +, Away spins - (mirrored). Watch if they look symmetric.\n")
    
    for i in range(200):
        # Apply velocity control
        p.setJointMotorControl2(
            sim.robot_uid, home_kicker_idx,
            p.VELOCITY_CONTROL,
            targetVelocity=10.0,  # positive
            force=100.0
        )
        p.setJointMotorControl2(
            sim.robot_uid, away_kicker_idx,
            p.VELOCITY_CONTROL,
            targetVelocity=-10.0,  # mirrored (negative)
            force=100.0
        )
        
        p.stepSimulation()
        time.sleep(1/240)
        
        if i % 50 == 0:
            home_pos = p.getJointState(sim.robot_uid, home_kicker_idx)[0]
            away_pos = p.getJointState(sim.robot_uid, away_kicker_idx)[0]
            print(f"  Step {i}: home={np.degrees(home_pos):+.0f}°, away={np.degrees(away_pos):+.0f}°")
    
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    print("""
    For the self-play mirroring to work correctly:
    
    1. SLIDER: When both policies output the same action value:
       - They should move to OPPOSITE Y positions (symmetric about Y=0)
       - This is achieved by negating the away slider action
    
    2. KICKER: When both policies output the same action value:
       - They should spin in OPPOSITE directions
       - This makes both appear to kick "forward" from their perspective
       - This is achieved by negating the away kicker velocity
    
    3. OBSERVATIONS: The away side's observations are mirrored:
       - Ball position: X and Y mirrored about table center
       - Ball velocity: X and Y negated
       - Slider position: Should be negated for away (CHECK YOUR CODE!)
    """)
    
    input("\nPress Enter to close...")


if __name__ == "__main__":
    main()