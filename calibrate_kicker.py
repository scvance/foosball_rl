"""
Find the correct zero position for each kicker so toes point straight down.

This script lets you manually adjust each kicker's position to find
the angle where the toes point straight down, then tells you what
offset to apply in the code or URDF.
"""

import numpy as np
import pybullet as p
import time
import sys

sys.path.insert(0, '/mnt/user-data/uploads')

from FoosballVersusEnv import FoosballVersusEnv


def interactive_find_zero(env):
    """
    Interactively find the zero position for each kicker.
    """
    print("\n" + "="*70)
    print("INTERACTIVE ZERO POSITION FINDER")
    print("="*70)
    print("""
Use keyboard input to adjust kicker positions until toes point straight down.
Enter a number (in degrees) to set the position, or:
  'h' = adjust home kicker
  'a' = adjust away kicker  
  'q' = quit and show results
  '+' or '-' = nudge by 5 degrees
  '++' or '--' = nudge by 1 degree
""")
    
    sim = env.sim
    
    home_offset = 0.0  # degrees
    away_offset = 0.0  # degrees
    
    current = 'h'  # Start with home
    
    while True:
        # Apply current offsets
        home_rad = np.radians(home_offset)
        away_rad = np.radians(away_offset)
        
        p.setJointMotorControl2(
            sim.robot_uid,
            sim.kicker_idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=0.0,
            force=1000.0,
        )
        p.resetJointState(sim.robot_uid, sim.kicker_idx, home_rad)
        
        p.setJointMotorControl2(
            sim.robot_uid,
            sim.opponent_kicker_idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=0.0,
            force=1000.0,
        )
        p.resetJointState(sim.robot_uid, sim.opponent_kicker_idx, away_rad)
        
        # Step to update visuals
        for _ in range(10):
            p.stepSimulation()
            time.sleep(0.01)
        
        print(f"\nCurrent: HOME={home_offset:+.1f}°, AWAY={away_offset:+.1f}°")
        print(f"Adjusting: {'HOME' if current == 'h' else 'AWAY'}")
        
        cmd = input("Enter command: ").strip().lower()
        
        if cmd == 'q':
            break
        elif cmd == 'h':
            current = 'h'
            print("Now adjusting HOME kicker")
        elif cmd == 'a':
            current = 'a'
            print("Now adjusting AWAY kicker")
        elif cmd == '+':
            if current == 'h':
                home_offset += 5
            else:
                away_offset += 5
        elif cmd == '-':
            if current == 'h':
                home_offset -= 5
            else:
                away_offset -= 5
        elif cmd == '++':
            if current == 'h':
                home_offset += 1
            else:
                away_offset += 1
        elif cmd == '--':
            if current == 'h':
                home_offset -= 1
            else:
                away_offset -= 1
        else:
            try:
                deg = float(cmd)
                if current == 'h':
                    home_offset = deg
                else:
                    away_offset = deg
            except ValueError:
                print("Invalid command")
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nTo make toes point straight down at 'zero':")
    print(f"  HOME kicker offset: {home_offset:+.1f}° = {np.radians(home_offset):+.4f} rad")
    print(f"  AWAY kicker offset: {away_offset:+.1f}° = {np.radians(away_offset):+.4f} rad")
    
    return home_offset, away_offset


def test_with_offsets(env, home_offset_deg, away_offset_deg, duration=8.0):
    """
    Test mirrored spinning with the discovered offsets applied.
    """
    print("\n" + "="*70)
    print("TEST: MIRRORED SPINNING WITH OFFSETS")
    print("="*70)
    print(f"Using offsets: HOME={home_offset_deg}°, AWAY={away_offset_deg}°")
    
    sim = env.sim
    policy_hz = env.policy_hz
    steps = int(duration * policy_hz)
    
    home_offset_rad = np.radians(home_offset_deg)
    away_offset_rad = np.radians(away_offset_deg)
    
    # Reset to offset positions first
    p.resetJointState(sim.robot_uid, sim.kicker_idx, home_offset_rad)
    p.resetJointState(sim.robot_uid, sim.opponent_kicker_idx, away_offset_rad)
    
    print("\nBoth kickers should now start with toes pointing down.")
    print("Applying mirrored velocity control...")
    
    for i in range(steps):
        t = i / steps
        
        # Oscillating velocity
        velocity = np.sin(2 * np.pi * t * 0.5) * 15.0  # rad/s
        
        p.setJointMotorControl2(
            sim.robot_uid,
            sim.kicker_idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=float(velocity),
            force=1000.0,
        )
        
        # Away: SAME velocity direction if they're oriented the same at zero
        # OR: OPPOSITE velocity if they're mirrored in the URDF
        # We need to test which one looks correct
        p.setJointMotorControl2(
            sim.robot_uid,
            sim.opponent_kicker_idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=float(-velocity),  # Try negated first
            force=1000.0,
        )
        
        for _ in range(env.steps_per_policy):
            p.stepSimulation()
            if env.render_mode == "human":
                time.sleep(0.0005)
        
        if i % 50 == 0:
            home_pos = p.getJointState(sim.robot_uid, sim.kicker_idx)[0]
            away_pos = p.getJointState(sim.robot_uid, sim.opponent_kicker_idx)[0]
            print(f"  vel={velocity:+6.1f} | home={np.degrees(home_pos):+7.1f}° | away={np.degrees(away_pos):+7.1f}°")


def show_current_positions(env):
    """Show current kicker positions at joint angle = 0."""
    print("\n" + "="*70)
    print("CURRENT ZERO POSITIONS")
    print("="*70)
    
    sim = env.sim
    
    # Set both to 0
    p.resetJointState(sim.robot_uid, sim.kicker_idx, 0.0)
    p.resetJointState(sim.robot_uid, sim.opponent_kicker_idx, 0.0)
    
    for _ in range(100):
        p.stepSimulation()
        time.sleep(0.01)
    
    print("\nBoth kickers set to joint angle = 0°")
    print("Look at the simulation - where are the toes pointing?")
    print("\nIf they're not pointing straight down, we need to find the offset.")


def sweep_to_find_down(env):
    """
    Slowly sweep each kicker to help find the 'down' position.
    """
    print("\n" + "="*70)
    print("SWEEP TO FIND DOWN POSITION")
    print("="*70)
    print("Slowly sweeping each kicker. Watch for when toes point straight down.")
    print("Note the angle displayed when that happens.")
    
    sim = env.sim
    
    print("\n--- Sweeping HOME kicker ---")
    for deg in range(-180, 181, 10):
        rad = np.radians(deg)
        p.resetJointState(sim.robot_uid, sim.kicker_idx, rad)
        p.resetJointState(sim.robot_uid, sim.opponent_kicker_idx, 0.0)
        
        for _ in range(20):
            p.stepSimulation()
            time.sleep(0.01)
        
        print(f"  HOME at {deg:+4d}°", end='\r')
    
    print("\n")
    home_down = input("At what angle (degrees) did HOME toes point straight down? ").strip()
    
    print("\n--- Sweeping AWAY kicker ---")
    for deg in range(-180, 181, 10):
        rad = np.radians(deg)
        p.resetJointState(sim.robot_uid, sim.kicker_idx, 0.0)
        p.resetJointState(sim.robot_uid, sim.opponent_kicker_idx, rad)
        
        for _ in range(20):
            p.stepSimulation()
            time.sleep(0.02)
        
        print(f"  AWAY at {deg:+4d}°", end='\r')
    
    print("\n")
    away_down = input("At what angle (degrees) did AWAY toes point straight down? ").strip()
    
    try:
        home_down = float(home_down)
        away_down = float(away_down)
        print(f"\n  HOME 'down' offset: {home_down}°")
        print(f"  AWAY 'down' offset: {away_down}°")
        return home_down, away_down
    except:
        print("Invalid input")
        return 0, 0


def print_code_fix(home_offset_deg, away_offset_deg):
    """Print the code changes needed to apply these offsets."""
    print("\n" + "="*70)
    print("CODE CHANGES NEEDED")
    print("="*70)
    
    home_rad = np.radians(home_offset_deg)
    away_rad = np.radians(away_offset_deg)
    
    print(f"""
In FoosballSimCore.py, add these offset constants:

    self.home_kicker_zero_offset = {home_rad:.4f}  # {home_offset_deg}° - toes down
    self.away_kicker_zero_offset = {away_rad:.4f}  # {away_offset_deg}° - toes down

Then when applying kicker targets, ADD the offset:

    # For home kicker:
    target_with_offset = target + self.home_kicker_zero_offset
    
    # For away kicker:
    target_with_offset = target + self.away_kicker_zero_offset

This makes 'target=0' mean 'toes pointing down' for both kickers.

---

ALTERNATIVELY, fix in the URDF by adjusting the joint origin rpy values.
This is cleaner but requires URDF editing.
""")


def main():
    print("="*70)
    print("KICKER ZERO POSITION CALIBRATION")
    print("="*70)
    # -58 degrees for home
    # -163 for away
    
    env = FoosballVersusEnv(
        render_mode="human",
        seed=42,
        policy_hz=200.0,
        sim_hz=1000,
        serve_side="random",
        max_episode_steps=100000,
    )
    
    obs, info = env.reset()
    
    try:
        # Show current zero positions
        show_current_positions(env)
        
        # input("\nPress Enter to sweep and find 'down' positions...")
        home_offset, away_offset = sweep_to_find_down(env)
        if home_offset != 0 or away_offset != 0:
            input("\nPress Enter to test with these offsets...")
            test_with_offsets(env, home_offset, away_offset)
        
        input("\nPress Enter for interactive fine-tuning...")
        home_offset, away_offset = interactive_find_zero(env)
        
        # Print code fix
        print_code_fix(home_offset, away_offset)
        
        # Final test
        input("\nPress Enter for final mirrored test with offsets...")
        test_with_offsets(env, home_offset, away_offset)
        
    except KeyboardInterrupt:
        print("\n\nCalibration interrupted.")
    finally:
        env.close()


if __name__ == "__main__":
    main()