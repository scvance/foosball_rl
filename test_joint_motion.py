"""
Diagnose and fix kicker joint limits to allow free spinning.

The kickers should be able to rotate continuously (360°+), not be
constrained to a limited range.
"""

import numpy as np
import pybullet as p
import time

from FoosballVersusEnv import FoosballVersusEnv


def diagnose_joint_type(env):
    """Check the joint type and limits from PyBullet."""
    print("\n" + "="*70)
    print("JOINT TYPE AND LIMITS DIAGNOSTIC")
    print("="*70)
    
    sim = env.sim
    
    # Joint types in PyBullet:
    # 0 = REVOLUTE (has limits)
    # 1 = PRISMATIC (has limits)
    # 2 = SPHERICAL
    # 3 = PLANAR
    # 4 = FIXED
    # 5 = CONTINUOUS (no limits, can spin freely)
    
    joint_type_names = {
        0: "REVOLUTE (limited rotation)",
        1: "PRISMATIC (limited translation)",
        2: "SPHERICAL",
        3: "PLANAR", 
        4: "FIXED",
        5: "CONTINUOUS (unlimited rotation)",
    }
    
    print("\n--- HOME KICKER ---")
    info = p.getJointInfo(sim.robot_uid, sim.kicker_idx)
    joint_name = info[1].decode()
    joint_type = info[2]
    lower_limit = info[8]
    upper_limit = info[9]
    max_force = info[10]
    max_velocity = info[11]
    
    print(f"  Joint name: {joint_name}")
    print(f"  Joint type: {joint_type} = {joint_type_names.get(joint_type, 'UNKNOWN')}")
    print(f"  Lower limit: {lower_limit:.4f} rad = {np.degrees(lower_limit):.1f}°")
    print(f"  Upper limit: {upper_limit:.4f} rad = {np.degrees(upper_limit):.1f}°")
    print(f"  Max force: {max_force}")
    print(f"  Max velocity: {max_velocity}")
    
    if joint_type == 0:  # REVOLUTE
        print(f"\n  *** THIS IS A REVOLUTE JOINT WITH LIMITS ***")
        print(f"  *** It can only rotate {np.degrees(upper_limit - lower_limit):.1f}° ***")
        print(f"  *** Need to change to CONTINUOUS in URDF or remove limits ***")
    
    print("\n--- AWAY KICKER ---")
    info = p.getJointInfo(sim.robot_uid, sim.opponent_kicker_idx)
    joint_name = info[1].decode()
    joint_type = info[2]
    lower_limit = info[8]
    upper_limit = info[9]
    max_force = info[10]
    max_velocity = info[11]
    
    print(f"  Joint name: {joint_name}")
    print(f"  Joint type: {joint_type} = {joint_type_names.get(joint_type, 'UNKNOWN')}")
    print(f"  Lower limit: {lower_limit:.4f} rad = {np.degrees(lower_limit):.1f}°")
    print(f"  Upper limit: {upper_limit:.4f} rad = {np.degrees(upper_limit):.1f}°")
    print(f"  Max force: {max_force}")
    print(f"  Max velocity: {max_velocity}")
    
    if joint_type == 0:
        print(f"\n  *** THIS IS A REVOLUTE JOINT WITH LIMITS ***")
    
    return joint_type


def test_velocity_control_spin(env, duration=10.0):
    """
    Test using VELOCITY_CONTROL instead of POSITION_CONTROL.
    This might allow continuous spinning even with a revolute joint.
    """
    print("\n" + "="*70)
    print("TEST: VELOCITY CONTROL FOR CONTINUOUS SPIN")
    print("="*70)
    print("Using velocity control to try to spin continuously.")
    print("Positive velocity = one direction, negative = other direction")
    
    sim = env.sim
    policy_hz = env.policy_hz
    steps = int(duration * policy_hz)
    
    print("\n--- Spinning both kickers with velocity control ---")
    
    for i in range(steps):
        t = i / steps
        
        # Oscillate velocity: spin one way, then the other
        velocity = np.sin(2 * np.pi * t * 0.3) * 10.0  # rad/s
        
        # Use VELOCITY_CONTROL instead of POSITION_CONTROL
        p.setJointMotorControl2(
            sim.robot_uid,
            sim.kicker_idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=float(velocity),
            force=1000.0,  # High force to overcome any resistance
        )
        
        # Away: opposite velocity for mirroring
        p.setJointMotorControl2(
            sim.robot_uid,
            sim.opponent_kicker_idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=float(-velocity),  # Negated for mirroring
            force=1000.0,
        )
        
        # Step simulation
        for _ in range(env.steps_per_policy):
            p.stepSimulation()
            if env.render_mode == "human":
                time.sleep(0.0005)
        
        if i % 40 == 0:
            home_pos = p.getJointState(sim.robot_uid, sim.kicker_idx)[0]
            away_pos = p.getJointState(sim.robot_uid, sim.opponent_kicker_idx)[0]
            home_vel = p.getJointState(sim.robot_uid, sim.kicker_idx)[1]
            away_vel = p.getJointState(sim.robot_uid, sim.opponent_kicker_idx)[1]
            print(f"  target_vel={velocity:+6.2f} | home: pos={np.degrees(home_pos):+7.1f}°, vel={home_vel:+6.2f} | "
                  f"away: pos={np.degrees(away_pos):+7.1f}°, vel={away_vel:+6.2f}")


def test_remove_limits_programmatically(env):
    """
    Try to remove joint limits programmatically using changeDynamics.
    Note: This may not work for all PyBullet versions.
    """
    print("\n" + "="*70)
    print("TEST: ATTEMPTING TO REMOVE LIMITS PROGRAMMATICALLY")
    print("="*70)
    
    sim = env.sim
    
    # Try setting very wide limits
    very_low = -100.0  # ~5700 degrees
    very_high = 100.0
    
    print(f"Attempting to set limits to [{very_low}, {very_high}] rad...")
    
    try:
        # This might not work - PyBullet doesn't always allow changing limits
        p.changeDynamics(
            sim.robot_uid,
            sim.kicker_idx,
            jointLowerLimit=very_low,
            jointUpperLimit=very_high,
        )
        p.changeDynamics(
            sim.robot_uid,
            sim.opponent_kicker_idx,
            jointLowerLimit=very_low,
            jointUpperLimit=very_high,
        )
        print("  changeDynamics call succeeded (but may not have effect)")
    except Exception as e:
        print(f"  changeDynamics failed: {e}")
    
    # Check if it worked
    info_home = p.getJointInfo(sim.robot_uid, sim.kicker_idx)
    info_away = p.getJointInfo(sim.robot_uid, sim.opponent_kicker_idx)
    
    print(f"\nAfter attempted change:")
    print(f"  Home kicker limits: [{info_home[8]:.2f}, {info_home[9]:.2f}]")
    print(f"  Away kicker limits: [{info_away[8]:.2f}, {info_away[9]:.2f}]")


def test_fast_spin(env, duration=8.0):
    """
    Test fast spinning to see the full behavior.
    """
    print("\n" + "="*70)
    print("TEST: FAST CONTINUOUS SPIN")
    print("="*70)
    print("Applying constant high velocity to spin continuously.")
    
    sim = env.sim
    policy_hz = env.policy_hz
    steps = int(duration * policy_hz)
    
    spin_velocity = 20.0  # rad/s - about 3 rotations per second
    
    print(f"\nSpinning at {spin_velocity} rad/s ({np.degrees(spin_velocity):.0f}°/s)")
    print("Home spins positive, Away spins negative (mirrored)")
    
    for i in range(steps):
        p.setJointMotorControl2(
            sim.robot_uid,
            sim.kicker_idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=spin_velocity,
            force=1000.0,
        )
        p.setJointMotorControl2(
            sim.robot_uid,
            sim.opponent_kicker_idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=-spin_velocity,  # Mirrored
            force=1000.0,
        )
        
        for _ in range(env.steps_per_policy):
            p.stepSimulation()
            if env.render_mode == "human":
                time.sleep(0.0005)
        
        if i % 50 == 0:
            home_pos = p.getJointState(sim.robot_uid, sim.kicker_idx)[0]
            away_pos = p.getJointState(sim.robot_uid, sim.opponent_kicker_idx)[0]
            # Convert to rotations
            home_rot = home_pos / (2 * np.pi)
            away_rot = away_pos / (2 * np.pi)
            print(f"  Home: {home_rot:+.2f} rotations | Away: {away_rot:+.2f} rotations")


def print_urdf_fix():
    """Print instructions for fixing the URDF."""
    print("\n" + "="*70)
    print("HOW TO FIX THE URDF FOR CONTINUOUS ROTATION")
    print("="*70)
    print("""
In your URDF file, find the kicker joint definitions. They probably look like:

    <joint name="kicker" type="revolute">
        <limit lower="-1.57" upper="1.57" effort="100" velocity="100"/>
        ...
    </joint>

Change the joint type from "revolute" to "continuous":

    <joint name="kicker" type="continuous">
        <!-- Remove the <limit> tag entirely, or keep only effort/velocity -->
        <limit effort="100" velocity="100"/>
        ...
    </joint>

Do the same for "opponent_kicker" joint.

The key changes:
1. type="revolute" → type="continuous"
2. Remove lower="" and upper="" from the <limit> tag
   (or remove the <limit> tag entirely if you don't need effort/velocity limits)

After making these changes, the kickers will be able to spin freely 360°+.
""")


def main():
    print("="*70)
    print("KICKER FREE SPIN DIAGNOSTIC AND FIX")
    print("="*70)
    
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
        # Diagnose current joint type
        joint_type = diagnose_joint_type(env)
        
        # Try to remove limits programmatically
        input("\nPress Enter to try removing limits programmatically...")
        test_remove_limits_programmatically(env)
        
        # Test velocity control
        input("\nPress Enter to test velocity control spinning...")
        test_velocity_control_spin(env)
        
        # Test fast spin
        input("\nPress Enter to test fast continuous spin...")
        obs, info = env.reset()
        test_fast_spin(env)
        
        # Print URDF fix instructions
        print_urdf_fix()
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print("""
If the joints are REVOLUTE (type=0) with limits, you need to:

1. PREFERRED: Edit the URDF file to change joint type to "continuous"
   This is the cleanest fix.

2. ALTERNATIVE: Use VELOCITY_CONTROL instead of POSITION_CONTROL
   This may allow spinning past limits but is messier.

3. WORKAROUND: If you must keep position control, the environment
   would need to track cumulative rotation and handle wrap-around,
   which is complex.

The URDF fix is strongly recommended.
""")
        
    except KeyboardInterrupt:
        print("\n\nDiagnostic interrupted.")
    finally:
        env.close()


if __name__ == "__main__":
    main()