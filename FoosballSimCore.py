# FoosballSimCore.py
#
# Updated:
# - Adds explicit velocity caps:
#     kicker: 170 rad/s
#     slider: 15 m/s (policy can choose 10–15; default is 15)
# - Overrides Bullet/URDF velocity caps by:
#     (1) setting maxVelocity in POSITION_CONTROL every step
#     (2) attempting changeDynamics(..., maxJointVelocity=...) (best-effort; ignored if unsupported)
# - Re-applies caps after resets so you don’t “lose” the override.
#
# Notes:
# - Revolute joint units: rad/s
# - Prismatic joint units: m/s

from __future__ import annotations

import math
import os
import random
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

import pybullet as p
import numpy as np


@dataclass(frozen=True)
class JointLimits:
    lower: float
    upper: float
    effort: float
    velocity: float


class _FoosballSimCore:
    """
    Low-level PyBullet sim core used by the Gymnasium env.
    - Concave trimesh table collider + optional thick outside wall catchers for anti-tunneling.
    - URDF provides slider + kicker joints.
    """

    def __init__(
        self,
        use_gui: bool,
        time_step: float,
        seed: Optional[int],
        base_pos=(0.0, 0.0, 0.10),
        urdf_filename: str = "onshape_robot/foosball_robot/foosball_robot.urdf",
        assets_dirname: str = "onshape_robot/foosball_robot/assets",
        table_mesh_filename: str = "main_body.stl",
        # physics
        num_substeps: int = 8,
        ball_restitution: float = 0.5,
        table_restitution: float = 0.5,
        wall_restitution: float = 0.5,
        ball_lateral_friction: float = 0.02,
        wall_lateral_friction: float = 0.02,
        add_wall_catchers: bool = False,
        # ---- NEW: velocity caps you asked for ----
        kicker_vel_cap: float = 170.0,   # rad/s
        slider_vel_cap: float = 15.0,    # m/s
    ):
        self.use_gui = use_gui
        self.dt = float(time_step)
        self.rng = random.Random(seed)

        self.num_substeps = int(num_substeps)

        self.ball_restitution = float(ball_restitution)
        self.table_restitution = float(table_restitution)
        self.wall_restitution = float(wall_restitution)
        self.ball_lateral_friction = float(ball_lateral_friction)
        self.wall_lateral_friction = float(wall_lateral_friction)
        self.add_wall_catchers = bool(add_wall_catchers)

        # caps
        self.kicker_vel_cap = float(kicker_vel_cap)
        self.slider_vel_cap = float(slider_vel_cap)

        self.client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.resetSimulation()
        p.setTimeStep(self.dt)
        p.setGravity(0, 0, -9.81)

        # Physics engine parameters (substeps + CCD friendliness)
        base_kwargs = dict(enableConeFriction=1, numSolverIterations=100)
        try:
            p.setPhysicsEngineParameter(**base_kwargs, numSubSteps=self.num_substeps)
        except TypeError:
            p.setPhysicsEngineParameter(**base_kwargs)

        try:
            p.setPhysicsEngineParameter(allowedCcdPenetration=0.0)
        except TypeError:
            pass

        try:
            p.setPhysicsEngineParameter(restitutionVelocityThreshold=0.0)
        except TypeError:
            pass

        # Paths
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.assets_dir = os.path.join(self.root_dir, assets_dirname)
        self.urdf_path = os.path.join(self.root_dir, urdf_filename)
        self.table_mesh_path = self._resolve_mesh_path(table_mesh_filename)

        p.setAdditionalSearchPath(self.root_dir)

        # Table transform
        self.base_pos = list(map(float, base_pos))
        self.base_orn = [0.0, 0.0, 0.0, 1.0]

        # Load URDF
        self.robot_uid = p.loadURDF(
            self.urdf_path,
            basePosition=self.base_pos,
            baseOrientation=self.base_orn,
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )

        # Joints
        self.joint_name_to_idx = self._get_joint_name_map()
        self.slider_idx = self._require_joint("slider")
        self.kicker_idx = self._require_joint("kicker")
        # Optional opponent goalie joints (update URDF to expose these names).
        self.opponent_slider_idx = self.joint_name_to_idx.get("opponent_slider")
        self.opponent_kicker_idx = self.joint_name_to_idx.get("opponent_kicker")

        self.slider_limits = self._get_limits(self.slider_idx)
        self.kicker_limits = self._get_limits(self.kicker_idx)

        self.opponent_slider_limits = (
            self._get_limits(self.opponent_slider_idx) if self.opponent_slider_idx is not None else None
        )
        self.opponent_kicker_limits = (
            self._get_limits(self.opponent_kicker_idx) if self.opponent_kicker_idx is not None else None
        )

        # Flip opponent kicker by 180 deg (URDF is oriented upside-down); apply as an offset in commands/observations.
        self.opponent_kicker_offset = math.pi if self.opponent_kicker_idx is not None else 0.0

        # Disable built-in motors (so we own control)
        self._disable_default_motors()

        # ---- NEW: attempt to override Bullet/URDF hard caps (best-effort) ----
        self._apply_joint_velocity_caps()

        # Ball
        self.ball_radius = 0.0125
        self.ball_mass = 0.028
        self.ball_id: Optional[int] = None

        # Disable URDF base collisions (in case main_body has <collision>)
        p.setCollisionFilterGroupMask(self.robot_uid, -1, 0, 0)

        # Table concave collider
        self.table_body_id = self._add_table_concave_collider(self.table_mesh_path)
        self.env_body_ids: List[int] = [self.table_body_id]

        # Prevent robot vs environment collisions
        self._disable_robot_vs_env_collisions(self.env_body_ids)

        # Infer bounds
        self.table_min, self.table_max = p.getAABB(self.table_body_id, -1)
        self.table_min = list(map(float, self.table_min))
        self.table_max = list(map(float, self.table_max))
        self.table_min_local = [self.table_min[i] - self.base_pos[i] for i in range(3)]
        self.table_max_local = [self.table_max[i] - self.base_pos[i] for i in range(3)]

        # Estimate play surface z using ray tests
        self.play_surface_z = self._estimate_play_surface_z(self.table_min, self.table_max)
        self.play_surface_z_local = self.play_surface_z - self.base_pos[2]

        # Goal rectangle (table-local): plane x = -0.275, y in [-0.09, 0.09], z in [0, 0.10]
        self.goal_rect_x = -0.275
        self.goal_rect_y_min = -0.09
        self.goal_rect_y_max = 0.09
        self.goal_rect_z_min = 0.00
        self.goal_rect_z_max = 0.10
        self.goalie_x = -0.172  # slightly in front of goal line 
        # away
        self.goal_rect_x_away = 0.275
        self.goal_rect_y_min_away = -0.09
        self.goal_rect_y_max_away = 0.09
        self.goal_rect_z_min_away = 0.00
        self.goal_rect_z_max_away = 0.10
        self.goalie_x_away = 0.172  # slightly in front of goal line

        # Per-shot bookkeeping
        self.ball_steps = 0
        self.initial_vx_local: Optional[float] = None

        # For goal detection crossing
        self._prev_ball_local_pos: Optional[Tuple[float, float, float]] = None

        # For block detection
        self._stopped_steps_home = 0
        self._stopped_steps_away = 0
        self._stopped_steps = 0  # kept for backward compatibility

        # Optional catcher walls
        self.wall_catcher_ids: List[int] = []
        if self.add_wall_catchers:
            self.wall_catcher_ids = self._add_boundary_wall_catchers()
            self.env_body_ids.extend(self.wall_catcher_ids)
            self._disable_robot_vs_env_collisions(self.wall_catcher_ids)

        # HUD / debug items (GUI only)
        self._score_text_id: Optional[int] = None
        self._score_anchor_world = self.table_local_to_world_pos(
            [self.goal_rect_x, 0.0, self.goal_rect_z_max + 0.14]
        )
        self._dbg_ball_id: Optional[int] = None
        self._dbg_player_id: Optional[int] = None
        self._dbg_aim_line_id: Optional[int] = None
        self._dbg_goal_point_id: Optional[int] = None
        self._est_pos_vis: Optional[np.ndarray] = None
        self._est_vel_vis: Optional[np.ndarray] = None
        self._reward_zone_body_id: Optional[int] = None
        self._reward_zone_vis_id: Optional[int] = None
        self._reward_zone_half_extents: Optional[Tuple[float, float, float]] = None
        self._player_anchor_body_id: Optional[int] = None
        self._player_anchor_vis_id: Optional[int] = None
        self._player_anchor_half_extents: Optional[Tuple[float, float, float]] = None
        self._goal_intercept_local: Optional[Tuple[float, float, float]] = None
        self._goal_intercept_dir: Optional[Tuple[float, float, float]] = None

    # ----------------------------
    # Core utilities
    # ----------------------------

    def close(self):
        try:
            p.disconnect(self.client)
        except Exception:
            pass

    def _resolve_mesh_path(self, mesh_filename: str) -> str:
        candidates = [
            os.path.join(self.assets_dir, mesh_filename),
            os.path.join(self.root_dir, mesh_filename),
        ]
        if os.path.isabs(mesh_filename):
            candidates.insert(0, mesh_filename)
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"Could not find mesh '{mesh_filename}'. Tried:\n" + "\n".join(f"  - {c}" for c in candidates)
        )

    def _get_joint_name_map(self) -> Dict[str, int]:
        return {p.getJointInfo(self.robot_uid, i)[1].decode(): i for i in range(p.getNumJoints(self.robot_uid))}

    def _require_joint(self, name: str) -> int:
        if name not in self.joint_name_to_idx:
            raise KeyError(f"Expected joint '{name}'. Found: {sorted(self.joint_name_to_idx.keys())}")
        return self.joint_name_to_idx[name]

    def _get_limits(self, joint_idx: int) -> JointLimits:
        info = p.getJointInfo(self.robot_uid, joint_idx)
        return JointLimits(lower=float(info[8]), upper=float(info[9]), effort=float(info[10]), velocity=float(info[11]))

    def _clamp_to_limits(self, value: float, limits: JointLimits) -> float:
        return float(max(limits.lower, min(limits.upper, value)))

    @property
    def _controlled_joint_indices(self) -> List[int]:
        idxs = [self.slider_idx, self.kicker_idx]
        if self.opponent_slider_idx is not None:
            idxs.append(self.opponent_slider_idx)
        if self.opponent_kicker_idx is not None:
            idxs.append(self.opponent_kicker_idx)
        return idxs

    def _disable_default_motors(self) -> None:
        for idx in self._controlled_joint_indices:
            p.setJointMotorControl2(self.robot_uid, idx, controlMode=p.VELOCITY_CONTROL, force=0)

    def _apply_joint_velocity_caps(self) -> None:
        """
        Best-effort override of Bullet's internal per-joint max joint velocity.
        This does NOT replace the need to pass maxVelocity=... each step in POSITION_CONTROL.
        """
        joint_caps: Dict[int, float] = {
            self.slider_idx: self.slider_vel_cap,
            self.kicker_idx: self.kicker_vel_cap,
        }
        if self.opponent_slider_idx is not None:
            joint_caps[self.opponent_slider_idx] = self.slider_vel_cap
        if self.opponent_kicker_idx is not None:
            joint_caps[self.opponent_kicker_idx] = self.kicker_vel_cap

        for jid, cap in joint_caps.items():
            try:
                p.changeDynamics(self.robot_uid, jid, maxJointVelocity=float(abs(cap)))
            except TypeError:
                # Some builds don't accept maxJointVelocity; ignore.
                pass

    def reset_robot(self) -> None:
        self._reset_robot_to_values(0.0, 0.0, 0.0, self.opponent_kicker_offset)

    def reset_robot_randomized(self, rng: random.Random | None = None) -> None:
        if rng is None:
            rng = self.rng
        def sample_in_limits(lim: JointLimits) -> float:
            return rng.uniform(lim.lower, lim.upper)

        slider_home = sample_in_limits(self.slider_limits)
        kicker_home = sample_in_limits(self.kicker_limits)

        if self.opponent_slider_limits is not None:
            slider_opp = sample_in_limits(self.opponent_slider_limits)
        else:
            slider_opp = 0.0
        if self.opponent_kicker_limits is not None:
            kicker_opp = sample_in_limits(self.opponent_kicker_limits) + self.opponent_kicker_offset
        else:
            kicker_opp = self.opponent_kicker_offset

        self._reset_robot_to_values(slider_home, kicker_home, slider_opp, kicker_opp)

    def _reset_robot_to_values(self, slider_home: float, kicker_home: float, slider_opp: float, kicker_opp: float) -> None:
        p.resetJointState(self.robot_uid, self.slider_idx, targetValue=slider_home)
        p.resetJointState(self.robot_uid, self.kicker_idx, targetValue=kicker_home)
        if self.opponent_slider_idx is not None:
            p.resetJointState(self.robot_uid, self.opponent_slider_idx, targetValue=slider_opp)
        if self.opponent_kicker_idx is not None:
            p.resetJointState(self.robot_uid, self.opponent_kicker_idx, targetValue=kicker_opp)

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
    # Transforms
    # ----------------------------

    def world_to_table_local_pos(self, world_pos) -> np.ndarray:
        if hasattr(p, "invertTransform") and hasattr(p, "multiplyTransforms"):
            inv_pos, inv_orn = p.invertTransform(self.base_pos, self.base_orn)
            local_pos, _ = p.multiplyTransforms(inv_pos, inv_orn, world_pos, (0, 0, 0, 1))
            return np.array(local_pos, dtype=np.float32)
        return np.array([world_pos[i] - self.base_pos[i] for i in range(3)], dtype=np.float32)

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
            wpos, _ = p.multiplyTransforms(self.base_pos, self.base_orn, local_pos, (0, 0, 0, 1))
            return tuple(wpos)
        return (local_pos[0] + self.base_pos[0], local_pos[1] + self.base_pos[1], local_pos[2] + self.base_pos[2])

    def table_local_vec_to_world(self, v_local) -> Tuple[float, float, float]:
        if hasattr(p, "getMatrixFromQuaternion"):
            m = p.getMatrixFromQuaternion(self.base_orn)
            vx = m[0] * v_local[0] + m[1] * v_local[1] + m[2] * v_local[2]
            vy = m[3] * v_local[0] + m[4] * v_local[1] + m[5] * v_local[2]
            vz = m[6] * v_local[0] + m[7] * v_local[1] + m[8] * v_local[2]
            return (vx, vy, vz)
        return (float(v_local[0]), float(v_local[1]), float(v_local[2]))

    # ----------------------------
    # Environment bodies
    # ----------------------------

    def _disable_robot_vs_env_collisions(self, env_body_ids: List[int]) -> None:
        for env_id in env_body_ids:
            for link_idx in range(-1, p.getNumJoints(self.robot_uid)):
                p.setCollisionFilterPair(env_id, self.robot_uid, -1, link_idx, enableCollision=0)

    def _add_table_concave_collider(self, mesh_path: str) -> int:
        if not hasattr(p, "GEOM_FORCE_CONCAVE_TRIMESH"):
            raise RuntimeError("PyBullet build does not expose GEOM_FORCE_CONCAVE_TRIMESH.")
        col_shape = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=mesh_path,
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
        )
        body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=-1,
            basePosition=self.base_pos,
            baseOrientation=self.base_orn,
        )
        p.changeDynamics(
            body_id,
            -1,
            lateralFriction=self.wall_lateral_friction,
            restitution=self.table_restitution,
            rollingFriction=0.0,
            spinningFriction=0.0,
            linearDamping=0.0,
            angularDamping=0.0,
        )
        return body_id

    def _add_boundary_wall_catchers(self) -> List[int]:
        x_min, y_min, _ = self.table_min_local
        x_max, y_max, _ = self.table_max_local
        z0 = self.play_surface_z_local

        thickness = 0.06
        height = 0.25
        outside = self.ball_radius + 0.01

        half_x = 0.5 * (x_max - x_min) + thickness
        half_z = 0.5 * height
        half_y_thick = 0.5 * thickness

        wall_shape_y = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half_x, half_y_thick, half_z])
        wall_shape_x = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[half_y_thick, 0.5 * (y_max - y_min) + thickness, half_z]
        )

        bodies = []

        # +Y catcher
        pos_local = [0.5 * (x_min + x_max), y_max + outside + half_y_thick, z0 + half_z]
        bodies.append(self._create_static_box(wall_shape_y, pos_local))

        # -Y catcher
        pos_local = [0.5 * (x_min + x_max), y_min - outside - half_y_thick, z0 + half_z]
        bodies.append(self._create_static_box(wall_shape_y, pos_local))

        # +X catcher
        pos_local = [x_max + outside + half_y_thick, 0.5 * (y_min + y_max), z0 + half_z]
        bodies.append(self._create_static_box(wall_shape_x, pos_local))

        for bid in bodies:
            p.changeDynamics(
                bid,
                -1,
                lateralFriction=self.wall_lateral_friction,
                restitution=self.wall_restitution,
                rollingFriction=0.0,
                spinningFriction=0.0,
                linearDamping=0.0,
                angularDamping=0.0,
            )
        return bodies

    def _create_static_box(self, collision_shape_id: int, pos_local) -> int:
        pos_world = self.table_local_to_world_pos(pos_local)
        bid = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=-1,
            basePosition=pos_world,
            baseOrientation=self.base_orn,
        )
        return bid

    # ----------------------------
    # Player state helpers
    # ----------------------------

    def get_joint_positions(self) -> Tuple[float, float]:
        slider_pos = float(p.getJointState(self.robot_uid, self.slider_idx)[0])
        kicker_pos = float(p.getJointState(self.robot_uid, self.kicker_idx)[0])
        return slider_pos, kicker_pos

    def get_joint_positions_and_vels(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        s_state = p.getJointState(self.robot_uid, self.slider_idx)
        k_state = p.getJointState(self.robot_uid, self.kicker_idx)
        return (float(s_state[0]), float(k_state[0])), (float(s_state[1]), float(k_state[1]))

    # ----------------------------
    # Action application
    # ----------------------------

    def apply_action_targets(
        self,
        slider_target: float,
        kicker_target: float,
        slider_vel_cap: Optional[float] = None,
        kicker_vel_cap: Optional[float] = None,
        slider_force: Optional[float] = None,
        kicker_force: Optional[float] = None,
    ):
        """
        POSITION_CONTROL with explicit velocity caps.
        We intentionally do NOT respect URDF velocity limits here (since you want to override Bullet/URDF caps).
        If you want to respect URDF limits, clamp here using min(cap, self.*_limits.velocity).
        """
        s_cap = self.slider_vel_cap if slider_vel_cap is None else float(abs(slider_vel_cap))
        k_cap = self.kicker_vel_cap if kicker_vel_cap is None else float(abs(kicker_vel_cap))

        # s_force = float(self.slider_limits.effort) if slider_force is None else float(abs(slider_force))
        # k_force = float(self.kicker_limits.effort) if kicker_force is None else float(abs(kicker_force))

        # Clamp targets to joint limits
        slider_target = self._clamp_to_limits(float(slider_target), self.slider_limits)
        kicker_target = self._clamp_to_limits(float(kicker_target), self.kicker_limits)
        # print("Applying action targets: slider_target =", slider_target, "kicker_target =", kicker_target)
        # print("With caps: s_cap =", s_cap, "k_cap =", k_cap, "s_force =", s_force, "k_force =", k_force)
        # print("Joint limits: slider_limits =", self.slider_limits, "kicker_limits =", self.kicker_limits)

        p.setJointMotorControl2(
            self.robot_uid,
            self.slider_idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=float(slider_target),
            # force=s_force,
            maxVelocity=float(s_cap),
        )
        p.setJointMotorControl2(
            self.robot_uid,
            self.kicker_idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=float(kicker_target),
            # force=k_force,
            maxVelocity=float(k_cap),
        )

    def apply_action_targets_dual(
        self,
        home_slider_target: float,
        home_kicker_target: float,
        away_slider_target: Optional[float],
        away_kicker_target: Optional[float],
        home_slider_vel_cap: Optional[float] = None,
        home_kicker_vel_cap: Optional[float] = None,
        away_slider_vel_cap: Optional[float] = None,
        away_kicker_vel_cap: Optional[float] = None,
        home_slider_force: Optional[float] = None,
        home_kicker_force: Optional[float] = None,
        away_slider_force: Optional[float] = None,
        away_kicker_force: Optional[float] = None,
    ):
        """Apply targets for both goalies; away targets are ignored if opponent joints are missing."""
        self.apply_action_targets(
            home_slider_target,
            home_kicker_target,
            slider_vel_cap=home_slider_vel_cap,
            kicker_vel_cap=home_kicker_vel_cap,
            slider_force=home_slider_force,
            kicker_force=home_kicker_force,
        )

        # Opponent slider
        if self.opponent_slider_idx is not None and away_slider_target is not None and self.opponent_slider_limits is not None:
            s_cap = self.slider_vel_cap if away_slider_vel_cap is None else float(abs(away_slider_vel_cap))
            s_force = float(self.opponent_slider_limits.effort) if away_slider_force is None else float(abs(away_slider_force))
            away_slider_target = self._clamp_to_limits(float(away_slider_target), self.opponent_slider_limits)

            p.setJointMotorControl2(
                self.robot_uid,
                self.opponent_slider_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(away_slider_target),
                force=float(s_force),
                maxVelocity=float(s_cap),
            )

        # Opponent kicker (apply offset into internal joint coordinate)
        if self.opponent_kicker_idx is not None and away_kicker_target is not None and self.opponent_kicker_limits is not None:
            k_cap = self.kicker_vel_cap if away_kicker_vel_cap is None else float(abs(away_kicker_vel_cap))
            k_force = float(self.opponent_kicker_limits.effort) if away_kicker_force is None else float(abs(away_kicker_force))

            away_kicker_target = float(away_kicker_target) + float(self.opponent_kicker_offset)
            away_kicker_target = self._clamp_to_limits(float(away_kicker_target), self.opponent_kicker_limits)

            p.setJointMotorControl2(
                self.robot_uid,
                self.opponent_kicker_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(away_kicker_target),
                force=float(k_force),
                maxVelocity=float(k_cap),
            )

    # ----------------------------
    # Simulation stepping
    # ----------------------------

    def step_sim(self, n_steps: int):
        for _ in range(int(n_steps)):
            p.stepSimulation()
            if self.ball_id is not None:
                self.ball_steps += 1
            if self.use_gui:
                self._update_debug_markers()

    # ----------------------------
    # Table floor estimation
    # ----------------------------

    def _estimate_play_surface_z(self, table_min, table_max) -> float:
        x_mid = 0.5 * (table_min[0] + table_max[0])
        y_mid = 0.5 * (table_min[1] + table_max[1])

        sample_points = [
            (x_mid, y_mid),
            (x_mid + 0.05, y_mid),
            (x_mid - 0.05, y_mid),
            (x_mid, y_mid + 0.05),
            (x_mid, y_mid - 0.05),
        ]
        z_start = table_max[2] + 0.5
        z_end = table_min[2] - 0.5

        best_hit_z = None
        for x, y in sample_points:
            hit = p.rayTest([x, y, z_start], [x, y, z_end])[0]
            hit_body, hit_fraction, hit_pos = hit[0], hit[2], hit[3]
            if hit_body == self.table_body_id and hit_fraction < 1.0:
                hz = float(hit_pos[2])
                if best_hit_z is None or hz > best_hit_z:
                    best_hit_z = hz

        if best_hit_z is not None:
            return best_hit_z
        return float(table_min[2] + 0.01)


    # ----------------------------
    # Ball spawning / shooting
    # ----------------------------


    def spawn_shot_random(
        self,
        speed_min: float,
        speed_max: float,
        bounce_prob: float,
        target: str = "home",
    ) -> Dict[str, float]:
        """
        Spawns ball and fires toward a goal.
        target: 'home' (default, legacy single-goalie), 'away', or 'random'.
        Returns dict with the chosen parameters (for debugging).
        """
        self.remove_ball()

        margin = self.ball_radius + 0.02
        x_min, y_min, _ = self.table_min_local
        x_max, y_max, _ = self.table_max_local

        # y spawn: ensure entire ball fits
        y_spawn = self.rng.uniform(y_min + margin, y_max - margin)

        tgt = target.lower()
        if tgt == "random":
            tgt = "home" if self.rng.random() < 0.5 else "away"
        if tgt == "home":
            x_goal = float(self.goal_rect_x)
        elif tgt == "away":
            x_goal = float(self.goal_rect_x_away)
        else:
            raise ValueError("target must be 'home', 'away', or 'random'")

        direction_to_goal = -1.0 if x_goal < 0.0 else 1.0

        if direction_to_goal < 0:
            # Shoot left: place ball on center/right side.
            x_safe_min = max(x_goal + 0.06, x_min + margin)
            x_safe_max = x_max - margin
            if x_safe_max < x_safe_min:
                x_safe_max = x_safe_min
            if self.rng.random() < 0.5 and (x_safe_min <= 0.0 <= x_safe_max):
                x_spawn = 0.0
            else:
                lo = max(0.05, x_safe_min)
                x_spawn = self.rng.uniform(lo, x_safe_max) if lo <= x_safe_max else x_safe_min
        else:
            # Shoot right: place ball on center/left side.
            x_safe_min = x_min + margin
            x_safe_max = min(x_goal - 0.06, x_max - margin)
            if x_safe_max < x_safe_min:
                x_safe_max = x_safe_min
            if self.rng.random() < 0.5 and (x_safe_min <= 0.0 <= x_safe_max):
                x_spawn = 0.0
            else:
                hi = min(-0.05, x_safe_max)
                if x_safe_min <= hi:
                    x_spawn = self.rng.uniform(x_safe_min, hi)
                else:
                    x_spawn = x_safe_max

        z_spawn = self.play_surface_z_local + self.ball_radius + 0.001

        y_target = self.rng.uniform(self.goal_rect_y_min + self.ball_radius, self.goal_rect_y_max - self.ball_radius)

        use_bounce = (self.rng.random() < bounce_prob)
        if use_bounce:
            y_wall = (y_max - margin) if (self.rng.random() < 0.5) else (y_min + margin)
            y_virtual = 2.0 * y_wall - y_target
            dx = x_goal - x_spawn
            dy = y_virtual - y_spawn
        else:
            dx = x_goal - x_spawn
            dy = y_target - y_spawn

        # Ensure direction points toward the intended goal.
        if direction_to_goal < 0 and dx >= -1e-6:
            dx = -0.1
        if direction_to_goal > 0 and dx <= 1e-6:
            dx = 0.1

        norm = math.sqrt(dx * dx + dy * dy)
        if norm < 1e-9:
            dx, dy, norm = -1.0, 0.0, 1.0

        dir_x = dx / norm
        dir_y = dy / norm
        speed = self.rng.uniform(speed_min, speed_max)
        v_local = np.array([speed * dir_x, speed * dir_y, 0.0], dtype=np.float32)

        self._spawn_ball_with_velocity_local(x_spawn, y_spawn, z_spawn, v_local)

        self.ball_steps = 0
        self.initial_vx_local = float(v_local[0])

        # init goal-crossing history
        bpos_w, _ = p.getBasePositionAndOrientation(self.ball_id)
        lpos = self.world_to_table_local_pos(bpos_w)
        self._prev_ball_local_pos = (float(lpos[0]), float(lpos[1]), float(lpos[2]))

        # init block counters
        self._stopped_steps_home = 0
        self._stopped_steps_away = 0
        self._stopped_steps = 0  # backward compatibility

        return {
            "x_spawn": float(x_spawn),
            "y_spawn": float(y_spawn),
            "y_target": float(y_target),
            "speed": float(speed),
            "use_bounce": float(use_bounce),
            "target_goal": tgt,
        }

    def _spawn_ball_with_velocity_local(self, x_local: float, y_local: float, z_local: float, v_local: np.ndarray):
        pos_world = self.table_local_to_world_pos([x_local, y_local, z_local])
        v_world = self.table_local_vec_to_world(v_local)

        col = p.createCollisionShape(p.GEOM_SPHERE, radius=self.ball_radius)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=self.ball_radius, rgbaColor=(0.9, 0.1, 0.1, 1.0))
        self.ball_id = p.createMultiBody(self.ball_mass, col, vis, pos_world)

        dyn_kwargs = dict(
            lateralFriction=self.ball_lateral_friction,
            rollingFriction=0.0,
            spinningFriction=0.0,
            restitution=self.ball_restitution,
            ccdSweptSphereRadius=self.ball_radius,
            linearDamping=0.0,
            angularDamping=0.0,
        )
        try:
            p.changeDynamics(self.ball_id, -1, ccdMotionThreshold=1e-4, **dyn_kwargs)
        except TypeError:
            p.changeDynamics(self.ball_id, -1, **dyn_kwargs)

        p.resetBaseVelocity(self.ball_id, linearVelocity=v_world, angularVelocity=(0, 0, 0))

    def update_scoreboard_text(self, goals: int, blocks: int, outs: int, highlight: Optional[str] = None) -> None:
        """
        Render a bold scoreboard above the goal mouth. No-op when GUI is disabled.
        highlight can be 'goal' | 'block' | 'out' to tint the text.
        """
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
        # Compact single-line scoreboard so counters stay tight together when zoomed in.
        text = f"G:{int(goals)}  B:{int(blocks)}  O:{int(outs)}"
        color = color_map.get(highlight, color_map[None])

        self._score_text_id = p.addUserDebugText(
            text,
            self._score_anchor_world,
            textColorRGB=color,
            textSize=1.2,
            lifeTime=0,
        )

    def _remove_debug_item(self, item_id_attr: str) -> None:
        iid = getattr(self, item_id_attr)
        if iid is None:
            return
        try:
            p.removeUserDebugItem(iid)
        except Exception:
            pass
        setattr(self, item_id_attr, None)

    def set_goal_intercept_debug(
        self,
        y_hit: Optional[float],
        z_hit: Optional[float],
        x_hit: Optional[float] = None,
        v_dir: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """
        Draws a small marker on the goal plane at the predicted intercept (GUI only).
        Pass None to clear.
        """
        if not self.use_gui:
            return

        if y_hit is None or z_hit is None:
            self._remove_debug_item("_dbg_goal_point_id")
            self._goal_intercept_local = None
            self._goal_intercept_dir = None
            return

        if x_hit is None:
            x_hit = self.goalie_x

        base_w = self.table_local_to_world_pos([x_hit, y_hit, z_hit])
        top_w = self.table_local_to_world_pos([x_hit, y_hit, z_hit + 0.03])
        self._dbg_goal_point_id = p.addUserDebugLine(
            base_w,
            top_w,
            lineColorRGB=(0.2, 1.0, 0.2),
            lineWidth=50,
            replaceItemUniqueId=self._dbg_goal_point_id if self._dbg_goal_point_id is not None else -1,
        )
        self._goal_intercept_local = (float(x_hit), float(y_hit), float(z_hit))
        if v_dir is not None:
            vx, vy, vz = map(float, v_dir)
            norm = math.sqrt(vx * vx + vy * vy + vz * vz)
            if norm > 1e-8:
                self._goal_intercept_dir = (vx / norm, vy / norm, vz / norm)
            else:
                self._goal_intercept_dir = None
        else:
            self._goal_intercept_dir = None

    def _remove_debug_box(self, body_attr: str, vis_attr: str, half_attr: str) -> None:
        body_id = getattr(self, body_attr)
        if body_id is not None:
            try:
                p.removeBody(body_id)
            except Exception:
                pass
        setattr(self, body_attr, None)
        setattr(self, vis_attr, None)
        setattr(self, half_attr, None)

    def _ensure_debug_box(
        self,
        *,
        body_attr: str,
        vis_attr: str,
        half_attr: str,
        center_local: Tuple[float, float, float],
        half_extents: Tuple[float, float, float],
        rgba_color: Tuple[float, float, float, float],
        orientation_world: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        """Create or reposition a translucent debug box (GUI only)."""
        if not self.use_gui:
            return

        half_extents = tuple(float(h) for h in half_extents)
        cached_half_extents = getattr(self, half_attr)
        if orientation_world is None:
            orientation_world = tuple(self.base_orn)
        else:
            orientation_world = tuple(float(o) for o in orientation_world)
        needs_rebuild = (
            getattr(self, body_attr) is None
            or cached_half_extents is None
            or any(abs(a - b) > 1e-6 for a, b in zip(cached_half_extents, half_extents))
        )

        center_world = self.table_local_to_world_pos(center_local)

        if needs_rebuild:
            self._remove_debug_box(body_attr, vis_attr, half_attr)
            vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba_color)
            body_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=vis_id,
                basePosition=center_world,
                baseOrientation=orientation_world,
            )
            setattr(self, vis_attr, vis_id)
            setattr(self, body_attr, body_id)
            setattr(self, half_attr, half_extents)
        else:
            try:
                p.resetBasePositionAndOrientation(getattr(self, body_attr), center_world, orientation_world)
            except Exception:
                # If resetting fails, rebuild on the next update.
                self._remove_debug_box(body_attr, vis_attr, half_attr)

    def _update_debug_markers(self) -> None:
        """
        Draws:
        - player center (vertical cyan line)
        - estimated ball local position (vertical yellow line)
        - estimated aim line from ball toward goal plane using estimated velocity
        """
        if not self.use_gui:
            return

        # Player center marker (always shown)
        player_c = self.get_player_center_local()
        player_start_w = self.table_local_to_world_pos(player_c)
        player_end_w = self.table_local_to_world_pos([player_c[0], player_c[1], player_c[2] + 0.05])
        self._dbg_player_id = p.addUserDebugLine(
            player_start_w,
            player_end_w,
            lineColorRGB=(0.2, 0.8, 1.0),
            lineWidth=3,
            replaceItemUniqueId=self._dbg_player_id if self._dbg_player_id is not None else -1,
        )

        # Reward window and measurement anchor visualization
        self._update_reward_debug_boxes(player_c)

        if self._est_pos_vis is None or self._est_vel_vis is None:
            self._remove_debug_item("_dbg_ball_id")
            self._remove_debug_item("_dbg_aim_line_id")
            return

        pos_l = self._est_pos_vis
        vel_l = self._est_vel_vis

        # Ball marker
        ball_start_w = self.table_local_to_world_pos(pos_l)
        ball_end_w = self.table_local_to_world_pos([pos_l[0], pos_l[1], pos_l[2] + 0.05])
        self._dbg_ball_id = p.addUserDebugLine(
            ball_start_w,
            ball_end_w,
            lineColorRGB=(1.0, 1.0, 0.2),
            lineWidth=3,
            replaceItemUniqueId=self._dbg_ball_id if self._dbg_ball_id is not None else -1,
        )

        # Aim line: intersection with goal plane using estimated velocity
        vx = float(vel_l[0])
        if abs(vx) > 1e-5:
            t = (float(self.goal_rect_x) - float(pos_l[0])) / vx
        else:
            t = -1.0

        if t > 0.0 and t < 5.0:
            y_hit = float(pos_l[1]) + float(vel_l[1]) * t
            z_hit = float(pos_l[2]) + float(vel_l[2]) * t
            aim_start_w = ball_start_w
            aim_end_w = self.table_local_to_world_pos([self.goal_rect_x, y_hit, z_hit])
            self._dbg_aim_line_id = p.addUserDebugLine(
                aim_start_w,
                aim_end_w,
                lineColorRGB=(0.98, 0.3, 0.98),
                lineWidth=2,
                replaceItemUniqueId=self._dbg_aim_line_id if self._dbg_aim_line_id is not None else -1,
            )
        else:
            self._remove_debug_item("_dbg_aim_line_id")

    def _get_player_toebox(self, home):
        """
        Returns the local position of the player's toe box (front of the foot).
        """
        if home:
            kicker_idx = self.kicker_idx
        else:
            if self.opponent_kicker_idx is None:
                raise RuntimeError("Opponent kicker index is not defined.")
            kicker_idx = self.opponent_kicker_idx
        
        link_state = p.getLinkState(self.robot_uid, kicker_idx, computeForwardKinematics=True)
        pos_w, orn_w = link_state[0], link_state[1]
        rot = p.getMatrixFromQuaternion(orn_w)
        # local down = (0, +1, 0); transform to world using rotation matrix (full player height).
        down_dir = (rot[1], rot[4], rot[7])
        player_height = 0.1  # meters
        toe_w = (
            pos_w[0] + player_height * down_dir[0] * 0.7,
            pos_w[1] + player_height * down_dir[1] * 0.7,
            pos_w[2] + player_height * down_dir[2] * 0.7,
        )
        toe_l = self.world_to_table_local_pos(toe_w)
        toe_half_extents = (0.015, 0.015, 0.015)
        return toe_l, toe_half_extents

    def _update_reward_debug_boxes(self, player_c: np.ndarray) -> None:
        """
        Draws:
        - Blue translucent box following the predicted ball path (goalie plane -> goal plane) where dense reward is possible.
        - Red translucent box centered on the point used to measure goalie position.
        """
        if not self.use_gui:
            return

        # Locate toes by walking a fixed local axis: use link local -Y (down the player body) with no runtime sign flipping.
        link_state = p.getLinkState(self.robot_uid, self.kicker_idx, computeForwardKinematics=True)
        pos_w, orn_w = link_state[0], link_state[1]
        rot = p.getMatrixFromQuaternion(orn_w)
        # local down = (0, +1, 0); transform to world using rotation matrix (full player height).
        down_dir = (rot[1], rot[4], rot[7])
        player_height = 0.1  # meters
        toe_w = (
            pos_w[0] + player_height * down_dir[0] * 0.7,
            pos_w[1] + player_height * down_dir[1] * 0.7,
            pos_w[2] + player_height * down_dir[2] * 0.7,
        )
        toe_l = self.world_to_table_local_pos(toe_w)
        toe_x, toe_y, toe_z = map(float, toe_l)

        y_center = toe_y
        y_half = float(self.get_player_y_halfwidth())
        denom = max(y_half + float(self.ball_radius), 1e-6)

        intercept = self._goal_intercept_local
        v_dir = self._goal_intercept_dir

        # Fallback: no intercept info or not moving toward our goal -> center under goalie at current toe y.
        if intercept is None or v_dir is None or v_dir[0] > -1e-4:
            intercept_y = y_center
            intercept_z = float(0.5 * (self.goal_rect_z_min + self.goal_rect_z_max))
            y_min = intercept_y - denom
            y_max = intercept_y + denom
            z_span = float(self.goal_rect_z_max - self.goal_rect_z_min)
            z_min = intercept_z - 0.5 * z_span - 0.02
            z_max = intercept_z + 0.5 * z_span + 0.05
            reward_x = float(self.goalie_x)
            reward_half_extents = (0.010, 1.1 * (y_max - y_min), 1.1 * (z_max - z_min))
            reward_center_local = (
                reward_x,
                0.5 * (y_min + y_max),
                0.5 * (z_min + z_max),
            )
            self._ensure_debug_box(
                body_attr="_reward_zone_body_id",
                vis_attr="_reward_zone_vis_id",
                half_attr="_reward_zone_half_extents",
                center_local=reward_center_local,
                half_extents=reward_half_extents,
                rgba_color=(0.1, 0.4, 1.0, 0.25),
                orientation_world=self.base_orn,
            )
        else:
            x_hit, intercept_y, intercept_z = intercept
            vx, vy, vz = v_dir
            # Half-widths perpendicular to the path
            y_min = intercept_y - denom
            y_max = intercept_y + denom
            z_span = float(self.goal_rect_z_max - self.goal_rect_z_min)
            z_min = intercept_z - 0.5 * z_span - 0.02
            z_max = intercept_z + 0.5 * z_span + 0.05
            half_y = 0.5 * 1.1 * (y_max - y_min)
            half_z = 0.5 * 1.1 * (z_max - z_min)

            # Build a box that starts at goalie_x and extends toward the goal line along v_dir.
            start_x = float(self.goalie_x)
            start_point = np.array([start_x, intercept_y, intercept_z], dtype=np.float32)
            dir_norm = np.array([vx, vy, vz], dtype=np.float32)

            # Find intersection with goal plane.
            if abs(dir_norm[0]) < 1e-6:
                length = 0.05
            else:
                t_goal = (float(self.goal_rect_x) - start_point[0]) / dir_norm[0]
                length = max(0.05, float(abs(t_goal)))

            end_point = start_point + dir_norm * length
            center_local = 0.5 * (start_point + end_point)
            half_x = 0.5 * length + 0.01
            half_extents = (half_x, float(half_y), float(half_z))

            # Orient box so its local +X points along dir_norm (yaw/pitch from direction).
            yaw = math.atan2(float(dir_norm[1]), float(dir_norm[0] if abs(dir_norm[0]) > 1e-9 else 1e-9))
            xy_mag = math.sqrt(float(dir_norm[0] * dir_norm[0] + dir_norm[1] * dir_norm[1]))
            pitch = math.atan2(float(dir_norm[2]), xy_mag)
            orientation_world = p.getQuaternionFromEuler((0.0, pitch, yaw))

            self._ensure_debug_box(
                body_attr="_reward_zone_body_id",
                vis_attr="_reward_zone_vis_id",
                half_attr="_reward_zone_half_extents",
                center_local=tuple(center_local.tolist()),
                half_extents=half_extents,
                rgba_color=(0.1, 0.4, 1.0, 0.25),
                orientation_world=orientation_world,
            )

        # Measure from the tip of the player's toes (link-based).
        anchor_half_extents = (0.015, 0.015, 0.015)
        anchor_center_local = (
            toe_x,
            toe_y,
            toe_z,
        )
        self._ensure_debug_box(
            body_attr="_player_anchor_body_id",
            vis_attr="_player_anchor_vis_id",
            half_attr="_player_anchor_half_extents",
            center_local=anchor_center_local,
            half_extents=anchor_half_extents,
            rgba_color=(0.1, 1.0, 0.2, 0.35),
            orientation_world=orn_w,
        )

    def set_estimated_ball_state(self, est_pos: Optional[np.ndarray], est_vel: Optional[np.ndarray]) -> None:
        """
        Accepts estimated ball state (table-local) for visualization. Pass None to clear.
        """
        if est_pos is None or est_vel is None:
            self._est_pos_vis = None
            self._est_vel_vis = None
            return
        self._est_pos_vis = np.array(est_pos, dtype=np.float32)
        self._est_vel_vis = np.array(est_vel, dtype=np.float32)

    # ----------------------------
    # Player state helpers
    # ----------------------------

    def get_opponent_joint_positions(self) -> Tuple[Optional[float], Optional[float]]:
        if self.opponent_slider_idx is None or self.opponent_kicker_idx is None:
            return None, None
        slider_pos = float(p.getJointState(self.robot_uid, self.opponent_slider_idx)[0])
        kicker_raw = float(p.getJointState(self.robot_uid, self.opponent_kicker_idx)[0])
        kicker_pos = kicker_raw - self.opponent_kicker_offset
        return slider_pos, kicker_pos

    def get_opponent_joint_positions_and_vels(self) -> Tuple[Tuple[Optional[float], Optional[float]], Tuple[Optional[float], Optional[float]]]:
        if self.opponent_slider_idx is None or self.opponent_kicker_idx is None:
            return (None, None), (None, None)
        s_state = p.getJointState(self.robot_uid, self.opponent_slider_idx)
        k_state = p.getJointState(self.robot_uid, self.opponent_kicker_idx)
        slider_pos = float(s_state[0])
        kicker_pos = float(k_state[0]) - self.opponent_kicker_offset
        return (slider_pos, kicker_pos), (float(s_state[1]), float(k_state[1]))

    def get_player_center_local(self) -> np.ndarray:
        # link index kicker_idx corresponds to the child link of joint "kicker"
        link_state = p.getLinkState(self.robot_uid, self.kicker_idx, computeForwardKinematics=True)
        pos_w = link_state[0]
        return self.world_to_table_local_pos(pos_w)

    def get_opponent_player_center_local(self) -> Optional[np.ndarray]:
        if self.opponent_kicker_idx is None:
            return None
        link_state = p.getLinkState(self.robot_uid, self.opponent_kicker_idx, computeForwardKinematics=True)
        pos_w = link_state[0]
        return self.world_to_table_local_pos(pos_w)

    def get_player_y_halfwidth(self) -> float:
        aabb_min, aabb_max = p.getAABB(self.robot_uid, self.kicker_idx)
        return max(0.5 * float(aabb_max[1] - aabb_min[1]), 1e-3)

    def get_opponent_y_halfwidth(self) -> Optional[float]:
        if self.opponent_kicker_idx is None:
            return None
        aabb_min, aabb_max = p.getAABB(self.robot_uid, self.opponent_kicker_idx)
        return max(0.5 * float(aabb_max[1] - aabb_min[1]), 1e-3)

    def get_ball_true_local_pos_vel(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.ball_id is None:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        pos_w, _ = p.getBasePositionAndOrientation(self.ball_id)
        lin_w, _ = p.getBaseVelocity(self.ball_id)
        pos_l = self.world_to_table_local_pos(pos_w)
        vel_l = self.world_vec_to_table_local(lin_w)
        return pos_l, vel_l

    # ----------------------------
    # Event detection (GOAL / BLOCK)
    # ----------------------------

    def check_ball_out_of_bounds(self) -> Tuple[bool, str]:
        """
        Returns (is_out, reason).
        Uses table AABB in TABLE-LOCAL frame and ensures the *entire ball* fits inside.
        Also catches falling below floor or flying way above the table.
        """
        if self.ball_id is None:
            return False, ""

        pos_l, _ = self.get_ball_true_local_pos_vel()
        x, y, z = float(pos_l[0]), float(pos_l[1]), float(pos_l[2])

        # Require the entire sphere fits inside the XY bounds
        # (ball center must be inside [min+R, max-R]).
        R = float(self.ball_radius)
        margin = 0.002  # small numerical margin

        x_min = float(self.table_min_local[0]) + R + margin
        x_max = float(self.table_max_local[0]) - R - margin
        y_min = float(self.table_min_local[1]) + R + margin
        y_max = float(self.table_max_local[1]) - R - margin

        if x < x_min or x > x_max:
            return True, "x_oob"
        if y < y_min or y > y_max:
            return True, "y_oob"

        # Fell through floor / numerical blow-up
        if z < float(self.play_surface_z_local) - 0.08:
            return True, "below_floor"

        # Flew above the table by a lot (AABB max z is usually top of walls/geometry)
        if z > float(self.table_max_local[2]) + 0.20:
            return True, "above_table"

        return False, ""

    def check_goal_crossing(self) -> bool:
        """Legacy single-goalie GOAL detection (home side)."""
        return self.check_goal_crossings_dual().get("home", False)

    def check_goal_crossings_dual(self) -> Dict[str, bool]:
        """
        Returns {"home": bool, "away": bool}. Uses a shared prev-pos cache.
        """
        if self.ball_id is None or self._prev_ball_local_pos is None:
            return {"home": False, "away": False}

        pos_l, _ = self.get_ball_true_local_pos_vel()
        x1, y1, z1 = float(pos_l[0]), float(pos_l[1]), float(pos_l[2])

        x0, y0, z0 = self._prev_ball_local_pos
        self._prev_ball_local_pos = (x1, y1, z1)

        home_hit = self._check_goal_crossing_for_plane(x0, y0, z0, x1, y1, z1, float(self.goal_rect_x), expect_increasing=False)
        away_hit = self._check_goal_crossing_for_plane(
            x0, y0, z0, x1, y1, z1, float(self.goal_rect_x_away), expect_increasing=True
        )

        return {"home": home_hit, "away": away_hit}

    def _check_goal_crossing_for_plane(
        self,
        x0: float,
        y0: float,
        z0: float,
        x1: float,
        y1: float,
        z1: float,
        x_goal: float,
        expect_increasing: bool,
    ) -> bool:
        """Detect crossing of a goal plane in the expected direction."""
        if expect_increasing:
            if not (x0 < x_goal and x1 >= x_goal):
                return False
        else:
            if not (x0 > x_goal and x1 <= x_goal):
                return False

        dx = x1 - x0
        if abs(dx) < 1e-12:
            return False

        t = (x_goal - x0) / dx
        if t < 0.0 or t > 1.0:
            return False

        y_hit = y0 + t * (y1 - y0)
        z_hit = z0 + t * (z1 - z0)

        y_min = float(self.goal_rect_y_min + self.ball_radius)
        y_max = float(self.goal_rect_y_max - self.ball_radius)
        z_min = float(self.goal_rect_z_min + self.ball_radius)
        z_max = float(self.goal_rect_z_max - self.ball_radius)

        return (y_min <= y_hit <= y_max) and (z_min <= z_hit <= z_max)

    def check_block_event(self) -> bool:
        """Legacy single-goalie BLOCK detection (home side)."""
        res = self.check_block_events_dual()
        self._stopped_steps = self._stopped_steps_home
        return res.get("home", False)

    def check_block_events_dual(self) -> Dict[str, bool]:
        """
        Returns {"home": bool, "away": bool}. Contacts are keyed to the kicker link of each goalie.
        """
        if self.ball_id is None:
            self._stopped_steps_home = 0
            self._stopped_steps_away = 0
            self._stopped_steps = 0
            return {"home": False, "away": False}

        pos_l, vel_l = self.get_ball_true_local_pos_vel()
        x, y, z = float(pos_l[0]), float(pos_l[1]), float(pos_l[2])
        vx = float(vel_l[0])
        speed = float(np.linalg.norm(vel_l))

        home_block = self._check_block_for_goal(
            x, y, z, vx, speed, float(self.goal_rect_x), self.kicker_idx, counter_attr="_stopped_steps_home"
        )
        away_block = self._check_block_for_goal(
            x, y, z, vx, speed, float(self.goal_rect_x_away), self.opponent_kicker_idx, counter_attr="_stopped_steps_away"
        )

        self._stopped_steps = self._stopped_steps_home
        return {"home": home_block, "away": away_block}

    def _check_block_for_goal(
        self,
        x: float,
        y: float,
        z: float,
        vx: float,
        speed: float,
        goal_x: float,
        kicker_idx: Optional[int],
        counter_attr: str,
    ) -> bool:
        defense_zone = abs(x - goal_x) < 0.15

        # Contact with player link (if provided)
        had_contact = False
        if kicker_idx is not None:
            contacts = p.getContactPoints(bodyA=self.ball_id, bodyB=self.robot_uid, linkIndexB=kicker_idx)
            had_contact = contacts is not None and len(contacts) > 0

        # Stop condition (sustained)
        steps = getattr(self, counter_attr)
        if defense_zone and speed < 0.1:
            steps += 1
        else:
            steps = 0
        setattr(self, counter_attr, steps)

        if steps >= 5:
            return True

        # Basic "reversal" heuristic: if ball is near goal plane and velocity now points away from goal.
        toward_goal = (goal_x < x and vx < -0.08) or (goal_x > x and vx > 0.08)
        if not toward_goal and abs(vx) > 0.1:
            return True

        # Contact-based block (immediate)
        if defense_zone and had_contact:
            return True

        return False

    # ----------------------------
    # Contacts
    # ----------------------------

    def had_contact_with_kicker(self, kicker_idx: Optional[int]) -> bool:
        if kicker_idx is None or self.ball_id is None:
            return False
        contacts = p.getContactPoints(bodyA=self.ball_id, bodyB=self.robot_uid, linkIndexB=kicker_idx)
        return contacts is not None and len(contacts) > 0
