# visualize_scripted_controls.py
#
# Runs a few scripted "policy-like" commands:
# - Slide left/right while holding kicker ready
# - Do two fast kicks (position step / snap) while sliding
# - Tracks and prints max joint velocities actually achieved
#
# Usage:
#   python visualize_scripted_controls.py
#
# Assumes FoosballSimCore.py is in the same directory and your URDF/assets paths are correct.

from __future__ import annotations

import time
import math
import numpy as np

from foosball_envs.FoosballSimCore import _FoosballSimCore


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def main():
    sim = _FoosballSimCore(
        use_gui=True,
        time_step=1.0 / 1000.0,
        seed=0,
        num_substeps=8,
        add_wall_catchers=False,
        kicker_vel_cap=170.0,   # rad/s
        slider_vel_cap=15.0,    # m/s
    )

    sim.reset_robot()

    # Convenience
    s_lim = sim.slider_limits
    k_lim = sim.kicker_limits

    # Pick “ready” and “strike” angles safely inside limits
    # (these are just generic; adjust once you know your actual range)
    k_ready = k_lim.lower + 0.10 * (k_lim.upper - k_lim.lower)
    k_strike = k_lim.lower + 0.90 * (k_lim.upper - k_lim.lower)

    # Slider positions
    s_left = s_lim.lower + 0.10 * (s_lim.upper - s_lim.lower)
    s_mid  = s_lim.lower + 0.50 * (s_lim.upper - s_lim.lower)
    s_right= s_lim.lower + 0.90 * (s_lim.upper - s_lim.lower)

    # Policy-like per-step caps (what you asked for)
    SLIDER_CAP = 15.0   # m/s
    KICKER_CAP = 170.0  # rad/s

    # Track maxima
    max_slider_v = 0.0
    max_kicker_v = 0.0

    # Script timeline (seconds)
    # We keep it explicit (piecewise) so it feels like an RL policy issuing targets each step.
    T = 1.20

    # Kick pulse windows (very short target changes; in POSITION_CONTROL this acts like a “snap”)
    kick1_t0, kick1_t1 = 0.35, 0.42
    kick2_t0, kick2_t1 = 0.80, 0.87

    # Slider motion phases
    # 0.0-0.30: go mid -> left
    # 0.30-0.60: left -> right (kick1 inside this)
    # 0.60-1.20: right -> mid (kick2 inside this)
    def slider_target(t):
        if t < 0.30:
            u = t / 0.30
            return (1 - u) * s_mid + u * s_left
        elif t < 0.60:
            u = (t - 0.30) / 0.30
            return (1 - u) * s_left + u * s_right
        else:
            u = (t - 0.60) / 0.60
            return (1 - u) * s_right + u * s_mid

    def kicker_target(t):
        # Default: hold ready
        # During each kick window: command strike briefly, then return to ready
        if kick1_t0 <= t < kick1_t1:
            return k_strike
        if kick2_t0 <= t < kick2_t1:
            return k_strike
        return k_ready

    n_steps = int(math.ceil(T / sim.dt))
    t0_wall = time.time()

    for k in range(n_steps):
        t = k * sim.dt

        s_tgt = clamp(slider_target(t), s_lim.lower, s_lim.upper)
        k_tgt = clamp(kicker_target(t), k_lim.lower, k_lim.upper)

        # Issue “policy” action
        sim.apply_action_targets(
            slider_target=s_tgt,
            kicker_target=k_tgt,
            slider_vel_cap=SLIDER_CAP,
            kicker_vel_cap=KICKER_CAP,
        )

        # Step physics
        sim.step_sim(1)

        # Read back achieved velocities
        (_, _), (s_v, k_v) = sim.get_joint_positions_and_vels()
        max_slider_v = max(max_slider_v, abs(s_v))
        max_kicker_v = max(max_kicker_v, abs(k_v))

        # Keep GUI realtime-ish (optional)
        # If you want maximum speed, delete this sleep.
        if sim.use_gui:
            # Match simulated dt roughly
            # (if your sim runs slower/faster, this keeps it watchable)
            elapsed = time.time() - t0_wall
            target = (k + 1) * sim.dt
            if target > elapsed:
                time.sleep(target - elapsed)

    print("\n=== Scripted Control Results ===")
    print(f"dt                 : {sim.dt:.6f} s")
    print(f"slider max |v|     : {max_slider_v:.4f} m/s   (cap {SLIDER_CAP} m/s)")
    print(f"kicker max |qd|    : {max_kicker_v:.4f} rad/s (cap {KICKER_CAP} rad/s)")
    print(f"kicker max rpm     : {max_kicker_v * 60.0 / (2.0 * math.pi):.1f} rpm")

    sim.close()


if __name__ == "__main__":
    main()
