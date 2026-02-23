# ShootoutSimCore.py
#
# PyBullet simulation core for the Shootout foosball robot.
#
# Robot: onshape_robot/shootout_robot/shootout_robot.urdf
# Joints:
#   - handle         (prismatic ±0.11 m):  home player lateral position
#   - paddle         (revolute, continuous): home player paddle rotation
#   - opponent_handle (prismatic ±0.11 m): away player lateral position
#   - opponent_paddle (revolute, continuous): away player paddle rotation
#
# Table geometry (table-local frame, metres):
#   X: goal planes at x = ±0.610  (entire back wall is the goal)
#   Y: side walls at   y = ±0.295
#   Z: curved floor — ~0.065 m at x=0, ~0.014 m at goal lines
#      goal opening spans z = 0 … +0.175 m
#
# Ball: ~40 mm diameter (radius = 0.020 m)
#
# Spawn modes:
#   spawn_ball_random_fire  — near the shooter's paddle, no initial velocity
#   spawn_ball_corner_serve — near corner (y≈−273 mm, x≈±600 mm), small x velocity

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pybullet as p


@dataclass(frozen=True)
class JointLimits:
    lower: float
    upper: float
    effort: float
    velocity: float


class _ShootoutSimCore:
    """
    Low-level PyBullet sim core for the Shootout robot.

    The URDF base link (angle_wall_ball_catch) contains all table geometry
    (tabletop, walls, rails, etc.) as collision meshes.  The ball collides with
    this base link plus the two paddle links.  Handle / gear-rod links have their
    collision disabled to avoid interfering with ball physics.
    """

    def __init__(
        self,
        use_gui: bool,
        time_step: float,
        seed: Optional[int],
        base_pos=(0.0, 0.0, 0.0),
        urdf_filename: str = "../onshape_robot/shootout_robot/shootout_robot.urdf",
        # ---- physics ----
        num_substeps: int = 8,
        ball_restitution: float = 0.30,
        wall_restitution: float = 0.85,
        paddle_restitution: float = 0.85,
        ball_lateral_friction: float = 0.05,
        wall_lateral_friction: float = 0.02,
        ball_linear_damping: float = 0.01,
        ball_angular_damping: float = 0.01,
        # ---- velocity caps ----
        handle_vel_cap: float = 10.0,   # m/s
        paddle_vel_cap: float = 20.0,   # rad/s
        # ---- paddle torques ----
        paddle_holding_torque: float = 10.0,   # Nm — hold when near-zero velocity
        paddle_spinning_torque: float = 15.0,  # Nm — active spin
    ):
        self.use_gui = use_gui
        self.dt = float(time_step)
        self.rng = random.Random(seed)
        self.num_substeps = int(num_substeps)

        self.ball_restitution = float(ball_restitution)
        self.wall_restitution = float(wall_restitution)
        self.paddle_restitution = float(paddle_restitution)
        self.ball_lateral_friction = float(ball_lateral_friction)
        self.wall_lateral_friction = float(wall_lateral_friction)
        self.ball_linear_damping = float(ball_linear_damping)
        self.ball_angular_damping = float(ball_angular_damping)

        self.handle_vel_cap = float(handle_vel_cap)
        self.paddle_vel_cap = float(paddle_vel_cap)
        self.paddle_holding_torque = float(paddle_holding_torque)
        self.paddle_spinning_torque = float(paddle_spinning_torque)

        # ---- Table constants (metres) ----
        self.goal_x = 0.610          # goal plane |x| — entire back wall is the goal
        self.table_y_half = 0.295    # half-width
        self.floor_z_center = 0.065  # floor height at x = 0
        self.floor_z_edge = 0.014    # floor height at goal lines (x = ±goal_x)
        self.goal_z_top = 0.175      # top of goal opening
        self.paddle_edge_x = 0.509   # leading-edge x of paddle at handle = 0

        # ---- Ball ----
        self.ball_radius = 0.020     # 40 mm diameter
        self.ball_mass = 0.056       # kg
        self.ball_id: Optional[int] = None

        # ---- Corner-serve parameters ----
        self.corner_serve_x = 0.600
        self.corner_serve_y = -0.273
        self.corner_serve_z = 0.055
        self.corner_serve_speed = 0.6   # m/s initial |vx|

        # ---- Connect to PyBullet ----
        self.client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.resetSimulation()
        p.setTimeStep(self.dt)
        p.setGravity(0, 0, -9.81)

        base_kwargs = dict(enableConeFriction=1, numSolverIterations=100)
        try:
            p.setPhysicsEngineParameter(**base_kwargs, numSubSteps=self.num_substeps)
        except TypeError:
            p.setPhysicsEngineParameter(**base_kwargs)

        for kwarg in [
            dict(allowedCcdPenetration=0.0),
            dict(restitutionVelocityThreshold=0.0),
        ]:
            try:
                p.setPhysicsEngineParameter(**kwarg)
            except TypeError:
                pass

        # ---- Paths ----
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.urdf_path = os.path.join(self.root_dir, urdf_filename)
        # Set search path to robot directory so package://assets/ resolves correctly
        self.robot_dir = os.path.dirname(os.path.abspath(self.urdf_path))
        p.setAdditionalSearchPath(self.robot_dir)

        # ---- Base transform ----
        self.base_pos = list(map(float, base_pos))
        self.base_orn = [0.0, 0.0, 0.0, 1.0]

        # ---- Load URDF ----
        self.robot_uid = p.loadURDF(
            self.urdf_path,
            basePosition=self.base_pos,
            baseOrientation=self.base_orn,
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )

        # ---- Joint / link indices ----
        self.joint_name_to_idx = self._get_joint_name_map()
        self.handle_idx = self._require_joint("handle")
        self.paddle_idx = self._require_joint("paddle")
        self.opponent_handle_idx = self.joint_name_to_idx.get("opponent_handle")
        self.opponent_paddle_idx = self.joint_name_to_idx.get("opponent_paddle")

        self.handle_limits = self._get_limits(self.handle_idx)
        self.opponent_handle_limits = (
            self._get_limits(self.opponent_handle_idx)
            if self.opponent_handle_idx is not None
            else None
        )

        # Paddle joints are revolute ±π but treated as fully continuous.
        self.paddle_is_continuous = True
        self.opponent_paddle_is_continuous = True

        # ---- Physics setup ----
        self._disable_default_motors()
        self._apply_joint_velocity_caps()
        self._configure_link_physics()

        # Disable collision on handle mechanism links (gear_rod, other_ball_handle).
        # Only paddles and the base-link table should interact with the ball.
        self._disable_handle_collisions()

        # ---- Bookkeeping ----
        self.ball_steps = 0
        self._prev_ball_local_pos: Optional[Tuple[float, float, float]] = None
        self._stopped_steps_home = 0
        self._stopped_steps_away = 0

        # ---- GUI debug ----
        self._score_text_id: Optional[int] = None
        self._score_anchor_world = self.table_local_to_world_pos(
            [0.0, 0.0, self.goal_z_top + 0.08]
        )

    # ----------------------------
    # Core utilities
    # ----------------------------

    def close(self) -> None:
        try:
            p.disconnect(self.client)
        except Exception:
            pass

    def _get_joint_name_map(self) -> Dict[str, int]:
        return {
            p.getJointInfo(self.robot_uid, i)[1].decode(): i
            for i in range(p.getNumJoints(self.robot_uid))
        }

    def _require_joint(self, name: str) -> int:
        if name not in self.joint_name_to_idx:
            raise KeyError(
                f"Expected joint '{name}'. Found: {sorted(self.joint_name_to_idx.keys())}"
            )
        return self.joint_name_to_idx[name]

    def _get_limits(self, joint_idx: int) -> JointLimits:
        info = p.getJointInfo(self.robot_uid, joint_idx)
        return JointLimits(
            lower=float(info[8]),
            upper=float(info[9]),
            effort=float(info[10]),
            velocity=float(info[11]),
        )

    def _clamp_to_limits(self, value: float, limits: JointLimits) -> float:
        return float(max(limits.lower, min(limits.upper, value)))

    @property
    def _controlled_joint_indices(self) -> List[int]:
        idxs = [self.handle_idx, self.paddle_idx]
        if self.opponent_handle_idx is not None:
            idxs.append(self.opponent_handle_idx)
        if self.opponent_paddle_idx is not None:
            idxs.append(self.opponent_paddle_idx)
        return idxs

    def _disable_default_motors(self) -> None:
        for idx in self._controlled_joint_indices:
            p.setJointMotorControl2(
                self.robot_uid, idx, controlMode=p.VELOCITY_CONTROL, force=0
            )

    def _apply_joint_velocity_caps(self) -> None:
        """Best-effort override of Bullet's per-joint max-velocity field."""
        caps: Dict[int, float] = {
            self.handle_idx: self.handle_vel_cap,
            self.paddle_idx: self.paddle_vel_cap,
        }
        if self.opponent_handle_idx is not None:
            caps[self.opponent_handle_idx] = self.handle_vel_cap
        if self.opponent_paddle_idx is not None:
            caps[self.opponent_paddle_idx] = self.paddle_vel_cap
        for jid, cap in caps.items():
            try:
                p.changeDynamics(self.robot_uid, jid, maxJointVelocity=float(abs(cap)))
            except TypeError:
                pass

    def _configure_link_physics(self) -> None:
        """Set per-link restitution and friction."""
        # Base link — the full table assembly (walls, floor, rails…)
        p.changeDynamics(
            self.robot_uid,
            -1,
            lateralFriction=self.wall_lateral_friction,
            restitution=self.wall_restitution,
            rollingFriction=0.0,
            spinningFriction=0.0,
            linearDamping=0.0,
            angularDamping=0.0,
        )
        # Paddle links — high restitution for a snappy kick
        for idx in [self.paddle_idx, self.opponent_paddle_idx]:
            if idx is not None:
                p.changeDynamics(
                    self.robot_uid,
                    idx,
                    lateralFriction=self.wall_lateral_friction,
                    restitution=self.paddle_restitution,
                    rollingFriction=0.0,
                    spinningFriction=0.0,
                )
        # Handle / gear-rod links — neutral (collision will be disabled separately)
        for idx in [self.handle_idx, self.opponent_handle_idx]:
            if idx is not None:
                p.changeDynamics(
                    self.robot_uid,
                    idx,
                    lateralFriction=0.0,
                    restitution=0.0,
                )

    def _disable_handle_collisions(self) -> None:
        """
        Disable collision for the handle mechanism links (gear_rod,
        other_ball_handle).  Only paddles + base link interact with the ball.
        """
        for idx in [self.handle_idx, self.opponent_handle_idx]:
            if idx is not None:
                p.setCollisionFilterGroupMask(self.robot_uid, idx, 0, 0)

    # ----------------------------
    # Coordinate transforms
    # ----------------------------

    def world_to_table_local_pos(self, world_pos) -> np.ndarray:
        if hasattr(p, "invertTransform") and hasattr(p, "multiplyTransforms"):
            inv_pos, inv_orn = p.invertTransform(self.base_pos, self.base_orn)
            local_pos, _ = p.multiplyTransforms(inv_pos, inv_orn, world_pos, (0, 0, 0, 1))
            return np.array(local_pos, dtype=np.float32)
        return np.array(
            [world_pos[i] - self.base_pos[i] for i in range(3)], dtype=np.float32
        )

    def world_vec_to_table_local(self, v_world) -> np.ndarray:
        if hasattr(p, "invertTransform") and hasattr(p, "getMatrixFromQuaternion"):
            _, inv_orn = p.invertTransform([0.0, 0.0, 0.0], self.base_orn)
            m = p.getMatrixFromQuaternion(inv_orn)
            vx = m[0] * v_world[0] + m[1] * v_world[1] + m[2] * v_world[2]
            vy = m[3] * v_world[0] + m[4] * v_world[1] + m[5] * v_world[2]
            vz = m[6] * v_world[0] + m[7] * v_world[1] + m[8] * v_world[2]
            return np.array([vx, vy, vz], dtype=np.float32)
        return np.array(v_world, dtype=np.float32)

    def table_local_to_world_pos(self, local_pos) -> Tuple[float, float, float]:
        if hasattr(p, "multiplyTransforms"):
            wpos, _ = p.multiplyTransforms(
                self.base_pos, self.base_orn, local_pos, (0, 0, 0, 1)
            )
            return tuple(wpos)
        return (
            local_pos[0] + self.base_pos[0],
            local_pos[1] + self.base_pos[1],
            local_pos[2] + self.base_pos[2],
        )

    def table_local_vec_to_world(self, v_local) -> Tuple[float, float, float]:
        if hasattr(p, "getMatrixFromQuaternion"):
            m = p.getMatrixFromQuaternion(self.base_orn)
            vx = m[0] * v_local[0] + m[1] * v_local[1] + m[2] * v_local[2]
            vy = m[3] * v_local[0] + m[4] * v_local[1] + m[5] * v_local[2]
            vz = m[6] * v_local[0] + m[7] * v_local[1] + m[8] * v_local[2]
            return (vx, vy, vz)
        return (float(v_local[0]), float(v_local[1]), float(v_local[2]))

    # ----------------------------
    # Reset
    # ----------------------------

    def reset_robot(self) -> None:
        self._reset_robot_to_values(0.0, 0.0, 0.0, 0.0)

    def reset_robot_randomized(self, rng: Optional[random.Random] = None) -> None:
        if rng is None:
            rng = self.rng
        handle_home = rng.uniform(self.handle_limits.lower, self.handle_limits.upper)
        paddle_home = rng.uniform(-math.pi, math.pi)
        handle_opp = (
            rng.uniform(self.opponent_handle_limits.lower, self.opponent_handle_limits.upper)
            if self.opponent_handle_limits is not None
            else 0.0
        )
        paddle_opp = rng.uniform(-math.pi, math.pi)
        self._reset_robot_to_values(handle_home, paddle_home, handle_opp, paddle_opp)

    def _reset_robot_to_values(
        self,
        handle_home: float,
        paddle_home: float,
        handle_opp: float,
        paddle_opp: float,
    ) -> None:
        p.resetJointState(self.robot_uid, self.handle_idx, targetValue=handle_home)
        p.resetJointState(self.robot_uid, self.paddle_idx, targetValue=paddle_home)
        if self.opponent_handle_idx is not None:
            p.resetJointState(
                self.robot_uid, self.opponent_handle_idx, targetValue=handle_opp
            )
        if self.opponent_paddle_idx is not None:
            p.resetJointState(
                self.robot_uid, self.opponent_paddle_idx, targetValue=paddle_opp
            )
        self._disable_default_motors()
        self._apply_joint_velocity_caps()

    def remove_ball(self) -> None:
        if self.ball_id is not None:
            try:
                p.removeBody(self.ball_id)
            except Exception:
                pass
        self.ball_id = None

    # ----------------------------
    # Floor geometry helpers
    # ----------------------------

    def floor_z_at_x(self, x_local: float) -> float:
        """
        Approximate curved-floor height at table-local x.
        Cosine curve: floor_z_center at x=0, floor_z_edge at x=±goal_x.
        """
        frac = min(1.0, abs(x_local) / self.goal_x)
        curve = (1.0 + math.cos(math.pi * frac)) / 2.0  # 1 at centre, 0 at edge
        return self.floor_z_edge + (self.floor_z_center - self.floor_z_edge) * curve

    def safe_spawn_z(self, x_local: float, margin: float = 0.005) -> float:
        """Return a ball-centre z safely above the floor at the given x."""
        return self.floor_z_at_x(x_local) + self.ball_radius + margin

    # ----------------------------
    # Ball spawning internals
    # ----------------------------

    def _spawn_ball_with_velocity_local(
        self,
        x_local: float,
        y_local: float,
        z_local: float,
        v_local: np.ndarray,
    ) -> None:
        pos_world = self.table_local_to_world_pos([x_local, y_local, z_local])
        v_world = self.table_local_vec_to_world(v_local)

        col = p.createCollisionShape(p.GEOM_SPHERE, radius=self.ball_radius)
        vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=self.ball_radius, rgbaColor=(0.9, 0.15, 0.15, 1.0)
        )
        self.ball_id = p.createMultiBody(self.ball_mass, col, vis, pos_world)

        dyn_kwargs = dict(
            lateralFriction=self.ball_lateral_friction,
            rollingFriction=0.005,
            spinningFriction=0.0,
            restitution=self.ball_restitution,
            ccdSweptSphereRadius=self.ball_radius,
            linearDamping=self.ball_linear_damping,
            angularDamping=self.ball_angular_damping,
        )
        try:
            p.changeDynamics(self.ball_id, -1, ccdMotionThreshold=1e-4, **dyn_kwargs)
        except TypeError:
            p.changeDynamics(self.ball_id, -1, **dyn_kwargs)

        p.resetBaseVelocity(
            self.ball_id, linearVelocity=v_world, angularVelocity=(0, 0, 0)
        )

    def _init_ball_bookkeeping(self) -> None:
        self.ball_steps = 0
        self._stopped_steps_home = 0
        self._stopped_steps_away = 0
        bpos_w, _ = p.getBasePositionAndOrientation(self.ball_id)
        lpos = self.world_to_table_local_pos(bpos_w)
        self._prev_ball_local_pos = (float(lpos[0]), float(lpos[1]), float(lpos[2]))

    # ----------------------------
    # Public spawn modes
    # ----------------------------

    def spawn_ball_random_fire(
        self,
        target: str = "away",
        random_y: bool = True,
    ) -> Dict[str, object]:
        """
        Spawn the ball in front of the shooter's paddle with no initial velocity.
        The agent then uses its paddle to kick the ball toward the target goal.

        target='away': home player fires — ball at x ≈ −480 mm
                       (home paddle leading edge at x ≈ −509 mm)
        target='home': away player fires — ball at x ≈ +480 mm
                       (away paddle leading edge at x ≈ +509 mm)
        """
        self.remove_ball()

        margin = self.ball_radius + 0.005
        y_min = -self.table_y_half + margin
        y_max = self.table_y_half - margin

        tgt = target.lower()
        clearance = 0.029   # gap between ball centre and paddle leading edge

        if tgt == "away":
            # Home fires: spawn just in front of home paddle (negative-x side)
            x_spawn = -(self.paddle_edge_x - clearance)
        elif tgt == "home":
            # Away fires: spawn just in front of away paddle (positive-x side)
            x_spawn = self.paddle_edge_x - clearance
        else:
            raise ValueError("target must be 'home' or 'away'")

        y_spawn = self.rng.uniform(y_min, y_max) if random_y else 0.0
        z_spawn = self.safe_spawn_z(x_spawn)

        self._spawn_ball_with_velocity_local(
            x_spawn, y_spawn, z_spawn, np.zeros(3, dtype=np.float32)
        )
        self._init_ball_bookkeeping()

        return {
            "x_spawn": float(x_spawn),
            "y_spawn": float(y_spawn),
            "z_spawn": float(z_spawn),
            "target": tgt,
        }

    def spawn_ball_corner_serve(
        self,
        side: str = "random",
        serve_speed: Optional[float] = None,
    ) -> Dict[str, object]:
        """
        Spawn ball near the corner at y ≈ −273 mm, x ≈ ±600 mm, z ≈ 55 mm
        with a small x velocity so the curved floor rolls it into play.

        side='home': spawns at x = −600 mm, vx = +serve_speed (toward away goal)
        side='away': spawns at x = +600 mm, vx = −serve_speed (toward home goal)
        side='random': randomly pick home or away
        """
        self.remove_ball()

        if side == "random":
            side = "home" if self.rng.random() < 0.5 else "away"

        speed = self.corner_serve_speed if serve_speed is None else float(serve_speed)

        if side == "home":
            x_spawn = -self.corner_serve_x
            vx = speed          # moves toward +x (away goal)
        else:
            x_spawn = self.corner_serve_x
            vx = -speed         # moves toward −x (home goal)

        y_spawn = self.corner_serve_y
        z_spawn = self.corner_serve_z
        vy = 0.05   # slight nudge toward y = 0 (table centre)

        v_local = np.array([vx, vy, 0.0], dtype=np.float32)
        self._spawn_ball_with_velocity_local(x_spawn, y_spawn, z_spawn, v_local)
        self._init_ball_bookkeeping()

        return {
            "x_spawn": float(x_spawn),
            "y_spawn": float(y_spawn),
            "z_spawn": float(z_spawn),
            "side": side,
            "vx": float(vx),
        }

    # ----------------------------
    # Action application
    # ----------------------------

    def _apply_paddle_control(
        self,
        paddle_idx: int,
        target_velocity: float,
    ) -> None:
        """Velocity-control a paddle joint; hold position when near-zero."""
        if abs(target_velocity) < 0.1:
            p.setJointMotorControl2(
                self.robot_uid,
                paddle_idx,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=0.0,
                force=self.paddle_holding_torque,
            )
        else:
            clamped = float(
                max(-self.paddle_vel_cap, min(self.paddle_vel_cap, target_velocity))
            )
            p.setJointMotorControl2(
                self.robot_uid,
                paddle_idx,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=clamped,
                force=self.paddle_spinning_torque,
            )

    def apply_action_targets(
        self,
        handle_target: float,
        paddle_velocity: float,
        handle_vel_cap: Optional[float] = None,
    ) -> None:
        """
        Apply control for the home side.
        handle_target  — target lateral position (prismatic, will be clamped)
        paddle_velocity — target angular velocity in rad/s
        """
        s_cap = self.handle_vel_cap if handle_vel_cap is None else float(abs(handle_vel_cap))
        handle_target = self._clamp_to_limits(float(handle_target), self.handle_limits)

        p.setJointMotorControl2(
            self.robot_uid,
            self.handle_idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=handle_target,
            maxVelocity=s_cap,
        )
        self._apply_paddle_control(self.paddle_idx, float(paddle_velocity))

    def apply_action_targets_dual(
        self,
        home_handle_target: float,
        home_paddle_velocity: float,
        away_handle_target: Optional[float],
        away_paddle_velocity: Optional[float],
        home_handle_vel_cap: Optional[float] = None,
        away_handle_vel_cap: Optional[float] = None,
    ) -> None:
        """Apply control for both sides simultaneously."""
        self.apply_action_targets(
            home_handle_target,
            home_paddle_velocity,
            handle_vel_cap=home_handle_vel_cap,
        )

        if (
            self.opponent_handle_idx is not None
            and away_handle_target is not None
            and self.opponent_handle_limits is not None
        ):
            s_cap = (
                self.handle_vel_cap
                if away_handle_vel_cap is None
                else float(abs(away_handle_vel_cap))
            )
            away_clamped = self._clamp_to_limits(
                float(away_handle_target), self.opponent_handle_limits
            )
            p.setJointMotorControl2(
                self.robot_uid,
                self.opponent_handle_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=away_clamped,
                maxVelocity=s_cap,
            )

        if self.opponent_paddle_idx is not None and away_paddle_velocity is not None:
            self._apply_paddle_control(
                self.opponent_paddle_idx, float(away_paddle_velocity)
            )

    # ----------------------------
    # Simulation stepping
    # ----------------------------

    def step_sim(self, n_steps: int) -> None:
        for _ in range(int(n_steps)):
            p.stepSimulation()
            if self.ball_id is not None:
                self.ball_steps += 1

    # ----------------------------
    # State queries
    # ----------------------------

    def get_joint_positions(self) -> Tuple[float, float]:
        h = float(p.getJointState(self.robot_uid, self.handle_idx)[0])
        k = float(p.getJointState(self.robot_uid, self.paddle_idx)[0])
        return h, k

    def get_joint_positions_and_vels(
        self,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        h = p.getJointState(self.robot_uid, self.handle_idx)
        k = p.getJointState(self.robot_uid, self.paddle_idx)
        return (float(h[0]), float(k[0])), (float(h[1]), float(k[1]))

    def get_opponent_joint_positions(
        self,
    ) -> Tuple[Optional[float], Optional[float]]:
        if self.opponent_handle_idx is None or self.opponent_paddle_idx is None:
            return None, None
        h = float(p.getJointState(self.robot_uid, self.opponent_handle_idx)[0])
        k = float(p.getJointState(self.robot_uid, self.opponent_paddle_idx)[0])
        return h, k

    def get_opponent_joint_positions_and_vels(
        self,
    ) -> Tuple[
        Tuple[Optional[float], Optional[float]],
        Tuple[Optional[float], Optional[float]],
    ]:
        if self.opponent_handle_idx is None or self.opponent_paddle_idx is None:
            return (None, None), (None, None)
        h = p.getJointState(self.robot_uid, self.opponent_handle_idx)
        k = p.getJointState(self.robot_uid, self.opponent_paddle_idx)
        return (float(h[0]), float(k[0])), (float(h[1]), float(k[1]))

    def get_ball_true_local_pos_vel(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.ball_id is None:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        pos_w, _ = p.getBasePositionAndOrientation(self.ball_id)
        lin_w, _ = p.getBaseVelocity(self.ball_id)
        return (
            self.world_to_table_local_pos(pos_w),
            self.world_vec_to_table_local(lin_w),
        )

    def get_player_center_local(self) -> np.ndarray:
        link_state = p.getLinkState(
            self.robot_uid, self.paddle_idx, computeForwardKinematics=True
        )
        return self.world_to_table_local_pos(link_state[0])

    def get_opponent_player_center_local(self) -> Optional[np.ndarray]:
        if self.opponent_paddle_idx is None:
            return None
        link_state = p.getLinkState(
            self.robot_uid, self.opponent_paddle_idx, computeForwardKinematics=True
        )
        return self.world_to_table_local_pos(link_state[0])

    def get_handle_y_halfwidth(self) -> float:
        """Approximate lateral half-width of the paddle link AABB."""
        aabb_min, aabb_max = p.getAABB(self.robot_uid, self.paddle_idx)
        return max(0.5 * float(aabb_max[1] - aabb_min[1]), 1e-3)

    # ----------------------------
    # Event detection
    # ----------------------------

    def check_goal_crossings_dual(self) -> Dict[str, bool]:
        """
        Returns {"home": bool, "away": bool}.

        Home goal at x = −goal_x, away goal at x = +goal_x.
        The entire back wall is the goal (full y and z opening).
        """
        if self.ball_id is None or self._prev_ball_local_pos is None:
            return {"home": False, "away": False}

        pos_l, _ = self.get_ball_true_local_pos_vel()
        x1, y1, z1 = float(pos_l[0]), float(pos_l[1]), float(pos_l[2])
        x0, y0, z0 = self._prev_ball_local_pos
        self._prev_ball_local_pos = (x1, y1, z1)

        home_hit = self._check_goal_crossing_for_plane(
            x0, y0, z0, x1, y1, z1,
            x_goal=-self.goal_x,
            expect_increasing=False,
        )
        away_hit = self._check_goal_crossing_for_plane(
            x0, y0, z0, x1, y1, z1,
            x_goal=self.goal_x,
            expect_increasing=True,
        )
        return {"home": home_hit, "away": away_hit}

    def _check_goal_crossing_for_plane(
        self,
        x0: float, y0: float, z0: float,
        x1: float, y1: float, z1: float,
        x_goal: float,
        expect_increasing: bool,
    ) -> bool:
        x_proximity = 0.05   # ball within 5 cm of goal plane can count

        if expect_increasing:
            crossed = x0 < x_goal <= x1
            past_goal = x1 > x_goal
        else:
            crossed = x0 > x_goal >= x1
            past_goal = x1 < x_goal

        near_goal = abs(x1 - x_goal) < x_proximity
        if not (crossed or (past_goal and near_goal)):
            return False

        dx = x1 - x0
        if abs(dx) < 1e-12:
            y_hit, z_hit = y1, z1
        else:
            t = max(0.0, min(1.0, (x_goal - x0) / dx))
            y_hit = y0 + t * (y1 - y0)
            z_hit = z0 + t * (z1 - z0)

        # Full table width is the goal; add 10 % margin
        y_margin = self.table_y_half * 0.10
        y_ok = abs(y_hit) <= self.table_y_half + y_margin
        z_ok = 0.0 <= z_hit <= self.goal_z_top

        return y_ok and z_ok

    def check_block_events_dual(self) -> Dict[str, bool]:
        """
        Returns {"home": bool, "away": bool}.
        A block occurs when the paddle contacts the ball in its defence zone,
        or when the ball comes to rest near the goal.
        """
        if self.ball_id is None:
            self._stopped_steps_home = 0
            self._stopped_steps_away = 0
            return {"home": False, "away": False}

        pos_l, vel_l = self.get_ball_true_local_pos_vel()
        x = float(pos_l[0])
        speed = float(np.linalg.norm(vel_l))

        home_block = self._check_block_for_goal(
            x, speed, -self.goal_x, self.paddle_idx, "_stopped_steps_home"
        )
        away_block = self._check_block_for_goal(
            x, speed, self.goal_x, self.opponent_paddle_idx, "_stopped_steps_away"
        )
        return {"home": home_block, "away": away_block}

    def _check_block_for_goal(
        self,
        x: float,
        speed: float,
        goal_x: float,
        paddle_idx: Optional[int],
        counter_attr: str,
    ) -> bool:
        defense_zone = abs(x - goal_x) < 0.15

        had_contact = False
        if paddle_idx is not None and self.ball_id is not None:
            contacts = p.getContactPoints(
                bodyA=self.ball_id, bodyB=self.robot_uid, linkIndexB=paddle_idx
            )
            had_contact = contacts is not None and len(contacts) > 0

        steps = getattr(self, counter_attr)
        if defense_zone and speed < 0.1:
            steps += 1
        else:
            steps = 0
        setattr(self, counter_attr, steps)

        if steps >= 5:
            return True
        if defense_zone and had_contact:
            return True
        return False

    def had_contact_with_paddle(self, paddle_idx: Optional[int]) -> bool:
        if paddle_idx is None or self.ball_id is None:
            return False
        contacts = p.getContactPoints(
            bodyA=self.ball_id, bodyB=self.robot_uid, linkIndexB=paddle_idx
        )
        return contacts is not None and len(contacts) > 0

    def check_ball_out_of_bounds(self) -> Tuple[bool, str]:
        """
        Returns (is_out, reason).
        Goal crossings are detected separately; this catches runaway states.
        """
        if self.ball_id is None:
            return False, ""

        pos_l, _ = self.get_ball_true_local_pos_vel()
        x, y, z = float(pos_l[0]), float(pos_l[1]), float(pos_l[2])

        # Way past goal lines (runaway / tunnelling)
        if abs(x) > self.goal_x + 0.25:
            return True, "x_oob"

        # Through a side wall
        if abs(y) > self.table_y_half + self.ball_radius + 0.02:
            return True, "y_oob"

        # Fell through floor
        if z < -0.05:
            return True, "below_floor"

        # Flew too high
        if z > self.goal_z_top + 0.40:
            return True, "above_table"

        return False, ""

    # ----------------------------
    # GUI scoreboard
    # ----------------------------

    def update_scoreboard_text(
        self,
        goals: int,
        blocks: int,
        outs: int,
        highlight: Optional[str] = None,
    ) -> None:
        if not self.use_gui:
            return
        if self._score_text_id is not None:
            try:
                p.removeUserDebugItem(self._score_text_id)
            except Exception:
                pass
        color_map = {
            "goal": (1.0, 0.2, 0.2),
            "block": (0.2, 1.0, 0.4),
            "out": (1.0, 0.85, 0.25),
            None: (1.0, 1.0, 1.0),
        }
        color = color_map.get(highlight, color_map[None])
        self._score_text_id = p.addUserDebugText(
            f"G:{int(goals)}  B:{int(blocks)}  O:{int(outs)}",
            self._score_anchor_world,
            textColorRGB=color,
            textSize=1.2,
            lifeTime=0,
        )
