"""
verify_paddle_collisions.py

Loads the shootout robot and overlays translucent coloured boxes that exactly
match the two collision bars declared in the URDF for each paddle.  As the
handles slide back and forth the boxes track the paddles in real time.

Run from the project root:
    python verify_paddle_collisions.py

Keys (PyBullet GUI):
    W  — toggle wireframe (shows the actual collision geometry PyBullet built)
    Mouse-drag / scroll — orbit / zoom camera
    Ctrl+C in terminal — quit
"""

import math
import sys
import time

import pybullet as p

sys.path.insert(0, "foosball_envs")
from ShootoutSimCore import _ShootoutSimCore

# ── Geometry constants matching the URDF collision origins ────────────────────

# Half-extents  (box size 130 × 10 × 80 mm)
HALF = [0.065, 0.005, 0.040]

# Local offset of each bar from its paddle link frame (same xyz as URDF collision)
# paddle (home):      visual/collision offset xyz = (0.507245, 0, -0.195903)
# other_paddle (away): offset xyz = (-0.507245, 0, -0.195903)
# Bars are at y = ±0.103 m from that base offset.
HOME_BASE_XZ  = ( 0.507245, -0.195903)   # (x, z) in link frame
AWAY_BASE_XZ  = (-0.507245, -0.195903)
HOME_RY       =  0.122173   # radians, rotation about local y (from URDF rpy)
AWAY_RY       = -0.122173
Y_OFFSETS     = [0.103, -0.103]

# Visual colours
RGBA_HOME = (0.0, 0.9, 1.0, 0.45)   # cyan  — home paddle
RGBA_AWAY = (1.0, 0.55, 0.0, 0.45)  # amber — away paddle


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_visual_box(half_extents, rgba):
    """Create a visual-only (no collision) box body."""
    vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba
    )
    body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vis,
        basePosition=[0, 0, -100],   # park far away until first update
    )
    return body


def bar_world_transform(link_state, base_xz, y_off, ry_rad):
    """
    Compute world position + orientation for a collision bar.

    link_state  — result of getLinkState(..., computeForwardKinematics=True)
    base_xz     — (x, z) offset in the link frame (from URDF collision origin)
    y_off       — y offset for this particular bar (±0.103 m)
    ry_rad      — rotation about local y matching URDF collision rpy
    """
    link_world_pos = link_state[4]   # world position of link frame
    link_world_orn = link_state[5]   # world orientation of link frame

    local_pos = [base_xz[0], y_off, base_xz[1]]
    local_orn = p.getQuaternionFromEuler([0.0, ry_rad, 0.0])

    world_pos, world_orn = p.multiplyTransforms(
        link_world_pos, link_world_orn,
        local_pos, local_orn,
    )
    return world_pos, world_orn


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sim = _ShootoutSimCore(
        use_gui=True,
        time_step=1.0 / 240,
        seed=0,
        num_substeps=1,
    )
    sim.reset_robot()

    # Enable wireframe so PyBullet's actual collision geometry is visible
    p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1)
    # Keep RGB rendering on alongside wireframe
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

    # Build tracking visual boxes
    home_boxes = [make_visual_box(HALF, RGBA_HOME) for _ in Y_OFFSETS]
    away_boxes = [make_visual_box(HALF, RGBA_AWAY) for _ in Y_OFFSETS]

    print("─" * 60)
    print("Paddle collision verifier")
    print("  Cyan  boxes = home  paddle collision bars")
    print("  Amber boxes = away  paddle collision bars")
    print("  Handles sweep slowly so you can inspect all positions.")
    print("  Press W in the PyBullet window to toggle wireframe.")
    print("  Ctrl+C in terminal to quit.")
    print("─" * 60)

    try:
        t = 0.0
        dt = 1.0 / 240
        while True:
            t += dt

            # Sweep handles sinusoidally so boxes move across their range
            h = 0.10 * math.sin(t * 0.4)
            sim.apply_action_targets_dual(h, 0.0, -h, 0.0)
            sim.step_sim(1)

            # Get current link-frame world transforms
            home_ls = p.getLinkState(
                sim.robot_uid, sim.paddle_idx, computeForwardKinematics=True
            )
            away_ls = p.getLinkState(
                sim.robot_uid, sim.opponent_paddle_idx, computeForwardKinematics=True
            )

            # Update home bar overlays
            for box_body, y_off in zip(home_boxes, Y_OFFSETS):
                wp, wo = bar_world_transform(home_ls, HOME_BASE_XZ, y_off, HOME_RY)
                p.resetBasePositionAndOrientation(box_body, wp, wo)

            # Update away bar overlays
            for box_body, y_off in zip(away_boxes, Y_OFFSETS):
                wp, wo = bar_world_transform(away_ls, AWAY_BASE_XZ, y_off, AWAY_RY)
                p.resetBasePositionAndOrientation(box_body, wp, wo)

            time.sleep(dt)

    except KeyboardInterrupt:
        pass
    finally:
        sim.close()


if __name__ == "__main__":
    main()
