import math
import os
import random
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

import pybullet as p


@dataclass(frozen=True)
class JointLimits:
    lower: float
    upper: float
    effort: float
    velocity: float


class FoosballSim:
    """Autoplay foosball sim with goal/block detection, CCD, and anti-tunneling wall catchers."""

    def __init__(
        self,
        use_gui: bool = True,
        time_step: float = 1.0 / 240.0,
        seed: Optional[int] = None,
        base_pos=(0.0, 0.0, 0.10),
        urdf_filename: str = "onshape_robot/foosball_robot/foosball_robot.urdf",
        assets_dirname: str = "onshape_robot/foosball_robot/assets",
        table_mesh_filename: str = "main_body.stl",
        # bounciness / specular-ish bounce
        ball_restitution: float = 0.5,
        table_restitution: float = 0.5,
        wall_restitution: float = 0.5,
        # lower friction -> more "mirror-like" bounce
        ball_lateral_friction: float = 0.02,
        wall_lateral_friction: float = 0.02,
        # autoplay parameters
        autoplay: bool = True,
        respawn_delay_s: float = 2.0,
        bounce_prob: float = 0.25,
        speed_min: float = 2.0,
        speed_max: float = 10.0,
        # anti-tunneling
        num_substeps: int = 8,
        add_wall_catchers: bool = True,
    ):
        self.use_gui = use_gui
        self.dt = float(time_step)
        self.rng = random.Random(seed)

        self.ball_restitution = float(ball_restitution)
        self.table_restitution = float(table_restitution)
        self.wall_restitution = float(wall_restitution)
        self.ball_lateral_friction = float(ball_lateral_friction)
        self.wall_lateral_friction = float(wall_lateral_friction)

        self.autoplay = bool(autoplay)
        self.respawn_delay_s = float(respawn_delay_s)
        self.bounce_prob = float(bounce_prob)
        self.speed_min = float(speed_min)
        self.speed_max = float(speed_max)
        self._next_shot_time = time.time()

        self.num_substeps = int(num_substeps)
        self.add_wall_catchers = bool(add_wall_catchers)

        self.client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.resetSimulation()

        # Base physics setup
        p.setTimeStep(self.dt)
        p.setGravity(0, 0, -9.81)

        # Try to set higher substeps and CCD-related params. Different builds accept different kwargs.
        # We set numSubSteps and allowedCcdPenetration where possible.
        self._configure_physics_engine()

        # Paths
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.assets_dir = os.path.join(self.root_dir, assets_dirname)
        self.urdf_path = os.path.join(self.root_dir, urdf_filename)
        self.table_mesh_path = self._resolve_mesh_path(table_mesh_filename)
        p.setAdditionalSearchPath(self.root_dir)

        # Table transform
        self.base_pos = list(map(float, base_pos))
        self.base_orn = [0.0, 0.0, 0.0, 1.0]

        # Load URDF (rod/player visuals + joints)
        self.robot_uid = p.loadURDF(
            self.urdf_path,
            basePosition=self.base_pos,
            baseOrientation=self.base_orn,
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )

        # Joints
        self.slider_idx, self.kicker_idx = self._get_joint_indices()
        self.slider_limits = self._get_limits(self.slider_idx)
        self.kicker_limits = self._get_limits(self.kicker_idx)
        self._disable_default_motors()

        # Ball params
        self.ball_radius = 0.0125
        self.ball_mass = 0.028

        # Safety: disable URDF base collisions (in case main_body has a <collision>)
        self._disable_urdf_base_collisions()

        # Add concave table collider
        self.table_body_id = self._add_table_concave_collider(
            mesh_path=self.table_mesh_path,
            base_pos_world=self.base_pos,
            base_orn_world=self.base_orn,
        )

        # Robot must NOT collide with the table collider
        self.env_body_ids: List[int] = [self.table_body_id]
        self._disable_robot_vs_env_collisions(self.env_body_ids)

        # Tune robot link dynamics (doesn't matter much if it never collides with env)
        self._tune_robot_dynamics()

        # Infer bounds
        self.table_min, self.table_max, self.play_surface_z, _, self.player_aabb = self._infer_table_frames()
        self.table_min_local = [self.table_min[i] - self.base_pos[i] for i in range(3)]
        self.table_max_local = [self.table_max[i] - self.base_pos[i] for i in range(3)]
        self.play_surface_z_local = self.play_surface_z - self.base_pos[2]

        # Goal rectangle (table-local)
        # home
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

        # Debug markers (goal corners)
        if self.use_gui:
            self._hud_anchor_world = [
                0.5 * (self.table_min[0] + self.table_max[0]),
                0.5 * (self.table_min[1] + self.table_max[1]),
                self.table_max[2] + 0.12,
            ]
            self._add_table_origin_marker()
            self._add_dark_blue_debug_points_local(
                [
                    (-0.275, 0.09, 0.00),
                    (-0.275, -0.09, 0.00),
                    (-0.275, 0.09, 0.10),
                    (-0.275, -0.09, 0.10),
                ]
            )
        else:
            self._hud_anchor_world = [0, 0, 0]

        # Add thick “catcher” walls OUTSIDE the table walls to prevent tunneling escape
        # (does not replace the rounded floor or the mesh, only acts as a safety net)
        self.wall_catcher_ids: List[int] = []
        if self.add_wall_catchers:
            self.wall_catcher_ids = self._add_boundary_wall_catchers()
            self.env_body_ids.extend(self.wall_catcher_ids)
            self._disable_robot_vs_env_collisions(self.wall_catcher_ids)

        # GUI dummies (ignored)
        if self.use_gui:
            p.addUserDebugParameter("angle_deg (ignored)", -180, 180, 0)
            p.addUserDebugParameter("power_mps (ignored)", 0, 10, 5)
            p.addUserDebugParameter("spawn_x_local (ignored)", -1, 1, 0.0)
            p.addUserDebugParameter("spawn_y_local (ignored)", -1, 1, 0.0)
            p.addUserDebugParameter("spawn_z_local (ignored)", 0, 1, 0.0)
            p.addUserDebugParameter("launch_toggle (ignored)", 0, 1, 0)
            p.addUserDebugParameter("stop_toggle (ignored)", 0, 1, 0)
            p.addUserDebugParameter("freeze_toggle (ignored)", 0, 1, 0)

        # Per-ball state
        self.ball_id: Optional[int] = None
        self.ball_steps = 0
        self._prev_ball_local_pos = None
        self._goal_printed_for_ball = False
        self._goal_text_id = None
        self._goal_text_expires_at = 0.0

        self._initial_vx_local = None
        self._block_printed_for_ball = False
        self._stopped_steps = 0
        self.block_speed_eps = 0.06
        self.block_vx_eps = 0.03
        self.block_stop_required_steps = 20
        self.block_dir_eps = 0.03
        self.block_reverse_vx_eps = 0.06
        self.block_min_steps_before_reverse = 3
        self._block_text_id = None
        self._block_text_expires_at = 0.0

        self._arrow_id = None

        # Camera
        if self.use_gui:
            self.cam_yaw = 90
            self.cam_pitch = -35
            self.cam_dist = 0.8
            self.cam_target = [
                0.5 * (self.table_min[0] + self.table_max[0]),
                0.5 * (self.table_min[1] + self.table_max[1]),
                self.table_max[2] + 0.10,
            ]
            self._apply_camera()

    # ----------------------------
    # Physics engine configuration
    # ----------------------------

    def _configure_physics_engine(self):
        # Make tunneling harder:
        # - numSubSteps increases internal stepping resolution
        # - allowedCcdPenetration=0 reduces allowed penetration in CCD (if supported)
        # - more solver iterations helps contact stability
        base_kwargs = dict(enableConeFriction=1, numSolverIterations=150)

        # Try with substeps
        try:
            p.setPhysicsEngineParameter(**base_kwargs, numSubSteps=self.num_substeps)
        except TypeError:
            p.setPhysicsEngineParameter(**base_kwargs)

        # Try allowedCcdPenetration (seen in PyBullet examples)
        try:
            p.setPhysicsEngineParameter(allowedCcdPenetration=0.0)
        except TypeError:
            pass

        # Some builds expose restitutionVelocityThreshold; if present, set low so restitution applies more often
        try:
            p.setPhysicsEngineParameter(restitutionVelocityThreshold=0.0)
        except TypeError:
            pass

    # ----------------------------
    # Path helpers
    # ----------------------------

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

    # ----------------------------
    # Joint helpers
    # ----------------------------

    def _get_joint_indices(self) -> Tuple[int, int]:
        names = {p.getJointInfo(self.robot_uid, i)[1].decode(): i for i in range(p.getNumJoints(self.robot_uid))}
        if "slider" not in names or "kicker" not in names:
            raise KeyError(f"Expected joints 'slider' and 'kicker'. Found: {sorted(names.keys())}")
        return names["slider"], names["kicker"]

    def _get_limits(self, joint_idx: int) -> JointLimits:
        info = p.getJointInfo(self.robot_uid, joint_idx)
        return JointLimits(lower=float(info[8]), upper=float(info[9]), effort=float(info[10]), velocity=float(info[11]))

    def _disable_default_motors(self) -> None:
        for idx in (self.slider_idx, self.kicker_idx):
            p.setJointMotorControl2(self.robot_uid, idx, controlMode=p.VELOCITY_CONTROL, force=0)

    def reset_robot(self) -> None:
        p.resetJointState(self.robot_uid, self.slider_idx, targetValue=0.0)
        p.resetJointState(self.robot_uid, self.kicker_idx, targetValue=0.0)
        self._disable_default_motors()

    # ----------------------------
    # Transforms
    # ----------------------------

    def _table_local_to_world(self, local_pos, local_orn=(0.0, 0.0, 0.0, 1.0)):
        if hasattr(p, "multiplyTransforms"):
            wpos, worn = p.multiplyTransforms(self.base_pos, self.base_orn, local_pos, local_orn)
            return wpos, worn
        return [self.base_pos[i] + local_pos[i] for i in range(3)], self.base_orn

    def _world_to_table_local(self, world_pos, world_orn=(0.0, 0.0, 0.0, 1.0)):
        if hasattr(p, "invertTransform") and hasattr(p, "multiplyTransforms"):
            inv_pos, inv_orn = p.invertTransform(self.base_pos, self.base_orn)
            local_pos, local_orn = p.multiplyTransforms(inv_pos, inv_orn, world_pos, world_orn)
            return local_pos, local_orn
        local_pos = [world_pos[i] - self.base_pos[i] for i in range(3)]
        return local_pos, world_orn

    def _world_vec_to_table_local(self, v_world):
        if hasattr(p, "invertTransform") and hasattr(p, "getMatrixFromQuaternion"):
            _, inv_orn = p.invertTransform([0.0, 0.0, 0.0], self.base_orn)
            m = p.getMatrixFromQuaternion(inv_orn)
            vx = m[0] * v_world[0] + m[1] * v_world[1] + m[2] * v_world[2]
            vy = m[3] * v_world[0] + m[4] * v_world[1] + m[5] * v_world[2]
            vz = m[6] * v_world[0] + m[7] * v_world[1] + m[8] * v_world[2]
            return [vx, vy, vz]
        return list(v_world)

    def _table_vec_to_world(self, v_local):
        if hasattr(p, "getMatrixFromQuaternion"):
            m = p.getMatrixFromQuaternion(self.base_orn)
            vx = m[0] * v_local[0] + m[1] * v_local[1] + m[2] * v_local[2]
            vy = m[3] * v_local[0] + m[4] * v_local[1] + m[5] * v_local[2]
            vz = m[6] * v_local[0] + m[7] * v_local[1] + m[8] * v_local[2]
            return [vx, vy, vz]
        return list(v_local)

    # ----------------------------
    # Environment colliders
    # ----------------------------

    def _disable_urdf_base_collisions(self) -> None:
        p.setCollisionFilterGroupMask(self.robot_uid, -1, 0, 0)

    def _add_table_concave_collider(self, mesh_path: str, base_pos_world, base_orn_world, restitution: float = None) -> int:
        if not hasattr(p, "GEOM_FORCE_CONCAVE_TRIMESH"):
            raise RuntimeError("Your PyBullet build does not expose GEOM_FORCE_CONCAVE_TRIMESH.")
        col_shape = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=mesh_path,
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
        )
        body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=-1,
            basePosition=base_pos_world,
            baseOrientation=base_orn_world,
        )

        # Bouncier + low friction for specular-ish rebounds
        p.changeDynamics(
            body_id,
            -1,
            lateralFriction=self.wall_lateral_friction,
            restitution=self.table_restitution if restitution is None else float(restitution),
            rollingFriction=0.0,
            spinningFriction=0.0,
            linearDamping=0.0,
            angularDamping=0.0,
        )
        return body_id

    def _disable_robot_vs_env_collisions(self, env_body_ids: List[int]) -> None:
        # Prevent rod/player collisions with environment catchers and table
        for env_id in env_body_ids:
            for link_idx in range(-1, p.getNumJoints(self.robot_uid)):
                p.setCollisionFilterPair(env_id, self.robot_uid, -1, link_idx, enableCollision=0)

    def _add_boundary_wall_catchers(self) -> List[int]:
        """Add thick walls slightly outside table AABB in Y (and +X) to prevent tunneling escape."""
        # Use local AABB
        x_min, y_min, _ = self.table_min_local
        x_max, y_max, _ = self.table_max_local
        z0 = self.play_surface_z_local

        thickness = 0.06  # 6cm thick catcher
        height = 0.25

        # Keep the catcher outside the mesh wall (so it doesn't change normal play)
        outside = self.ball_radius + 0.01

        # Dimensions
        half_x = 0.5 * (x_max - x_min) + thickness
        half_z = 0.5 * height
        half_y_thick = 0.5 * thickness

        # A single box shape for y-walls
        wall_shape_y = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half_x, half_y_thick, half_z])
        # Back wall (+x) catcher
        wall_shape_x = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half_y_thick, 0.5 * (y_max - y_min) + thickness, half_z])

        bodies = []

        # +Y catcher (outside)
        pos_local = [0.5 * (x_min + x_max), y_max + outside + half_y_thick, z0 + half_z]
        pos_world, _ = self._table_local_to_world(pos_local)
        bodies.append(p.createMultiBody(0.0, wall_shape_y, -1, pos_world, self.base_orn))

        # -Y catcher (outside)
        pos_local = [0.5 * (x_min + x_max), y_min - outside - half_y_thick, z0 + half_z]
        pos_world, _ = self._table_local_to_world(pos_local)
        bodies.append(p.createMultiBody(0.0, wall_shape_y, -1, pos_world, self.base_orn))

        # +X catcher (outside) to stop “back wall” tunneling
        pos_local = [x_max + outside + half_y_thick, 0.5 * (y_min + y_max), z0 + half_z]
        pos_world, _ = self._table_local_to_world(pos_local)
        bodies.append(p.createMultiBody(0.0, wall_shape_x, -1, pos_world, self.base_orn))

        # Dynamics for catchers: bouncy + low friction
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

    def _tune_robot_dynamics(self) -> None:
        p.changeDynamics(self.robot_uid, -1, lateralFriction=0.6, restitution=0.05)
        for link_idx in range(p.getNumJoints(self.robot_uid)):
            p.changeDynamics(self.robot_uid, link_idx, lateralFriction=0.6, restitution=0.05)

    # ----------------------------
    # Table inference
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

    def _infer_table_frames(self):
        table_aabb = p.getAABB(self.table_body_id, -1)
        player_aabb = p.getAABB(self.robot_uid, self.kicker_idx)
        table_min = list(map(float, table_aabb[0]))
        table_max = list(map(float, table_aabb[1]))
        play_surface_z = self._estimate_play_surface_z(table_min, table_max)
        ball_spawn_z = play_surface_z + self.ball_radius + 0.001
        return table_min, table_max, play_surface_z, ball_spawn_z, player_aabb

    # ----------------------------
    # Debug markers / HUD
    # ----------------------------

    def _add_table_origin_marker(self) -> None:
        dark_blue = (0.0, 0.0, 0.55, 1.0)
        r = 0.012
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=dark_blue)
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis,
            basePosition=self.base_pos,
            baseOrientation=self.base_orn,
        )
        p.addUserDebugLine(
            self.base_pos,
            [self.base_pos[0], self.base_pos[1], self.base_pos[2] + 0.06],
            [dark_blue[0], dark_blue[1], dark_blue[2]],
            lineWidth=4,
        )

    def _add_dark_blue_debug_points_local(self, offsets_m, radius=0.012):
        dark_blue = (0.0, 0.0, 0.55, 1.0)
        vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=dark_blue)
        for (ox, oy, oz) in offsets_m:
            wpos, _ = self._table_local_to_world([ox, oy, oz])
            p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=vis_id,
                basePosition=wpos,
                baseOrientation=[0, 0, 0, 1],
            )

    def _update_ball_local_hud(self):
        if not self.use_gui:
            return
        if self.ball_id is None:
            text = "ball_local: (no ball)"
        else:
            bpos, born = p.getBasePositionAndOrientation(self.ball_id)
            lpos, _ = self._world_to_table_local(bpos, born)
            text = f"ball_local (m): x={lpos[0]: .3f}, y={lpos[1]: .3f}, z={lpos[2]: .3f}"
        try:
            self._ball_local_text_id = p.addUserDebugText(
                text,
                self._hud_anchor_world,
                textColorRGB=[1, 1, 1],
                textSize=1.2,
                replaceItemUniqueId=getattr(self, "_ball_local_text_id", -1),
            )
        except TypeError:
            # older builds: re-add
            try:
                if getattr(self, "_ball_local_text_id", None) is not None:
                    p.removeUserDebugItem(self._ball_local_text_id)
            except Exception:
                pass
            self._ball_local_text_id = p.addUserDebugText(text, self._hud_anchor_world, textColorRGB=[1, 1, 1], textSize=1.2)

    # ----------------------------
    # Spawn + shoot (autoplay)
    # ----------------------------

    def _spawn_ball_with_velocity_local(self, x_local: float, y_local: float, z_local: float, v_local):
        if self.ball_id is not None:
            try:
                p.removeBody(self.ball_id)
            except Exception:
                pass
            self.ball_id = None

        wpos, _ = self._table_local_to_world([x_local, y_local, z_local])
        v_world = self._table_vec_to_world(v_local)

        col = p.createCollisionShape(p.GEOM_SPHERE, radius=self.ball_radius)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=self.ball_radius, rgbaColor=(0.9, 0.1, 0.1, 1.0))
        self.ball_id = p.createMultiBody(self.ball_mass, col, vis, wpos)

        # CCD: always set swept sphere radius; motion threshold if supported
        dyn_kwargs = dict(
            lateralFriction=self.ball_lateral_friction,
            rollingFriction=0.0,
            spinningFriction=0.0,
            restitution=self.ball_restitution,
            ccdSweptSphereRadius=self.ball_radius,  # critical
            linearDamping=0.0,
            angularDamping=0.0,
        )
        try:
            # some builds support ccdMotionThreshold
            p.changeDynamics(self.ball_id, -1, ccdMotionThreshold=1e-4, **dyn_kwargs)
        except TypeError:
            p.changeDynamics(self.ball_id, -1, **dyn_kwargs)

        p.resetBaseVelocity(self.ball_id, linearVelocity=v_world, angularVelocity=[0, 0, 0])

        self.ball_steps = 0
        self._goal_printed_for_ball = False
        self._block_printed_for_ball = False
        self._stopped_steps = 0

        bpos_w, born_w = p.getBasePositionAndOrientation(self.ball_id)
        lpos, _ = self._world_to_table_local(bpos_w, born_w)
        self._prev_ball_local_pos = (float(lpos[0]), float(lpos[1]), float(lpos[2]))
        self._initial_vx_local = float(v_local[0])

        if self.use_gui:
            self._update_ball_local_hud()

    def _spawn_and_shoot_random(self):
        # Make sure the *entire ball* fits: center must be at least (ball_radius + margin) away from edges
        margin = self.ball_radius + 0.02

        x_min, y_min, _ = self.table_min_local
        x_max, y_max, _ = self.table_max_local

        # spawn inside playable region
        y_spawn = self.rng.uniform(y_min + margin, y_max - margin)

        # x spawn: either 0 (if inside safe region) or in the positive-x half
        x_goal = self.goal_rect_x
        x_safe_min = max(x_goal + 0.06, x_min + margin)
        x_safe_max = x_max - margin
        if x_safe_max < x_safe_min:
            x_safe_max = x_safe_min

        if self.rng.random() < 0.5 and (x_safe_min <= 0.0 <= x_safe_max):
            x_spawn = 0.0
        else:
            # "other half": favor positive-x region
            lo = max(0.05, x_safe_min)
            x_spawn = self.rng.uniform(lo, x_safe_max) if lo <= x_safe_max else self.rng.uniform(x_safe_min, x_safe_max)

        z_spawn = self.play_surface_z_local + self.ball_radius + 0.001

        # Choose target y inside goal opening (minus radius so it fits)
        y_target = self.rng.uniform(self.goal_rect_y_min + self.ball_radius, self.goal_rect_y_max - self.ball_radius)

        # Bounce shots: mirror across y-wall (specular)
        use_bounce = (self.rng.random() < self.bounce_prob)
        if use_bounce:
            # pick which wall to bounce off (use inner wall approx)
            y_wall = (y_max - margin) if (self.rng.random() < 0.5) else (y_min + margin)
            y_virtual = 2.0 * y_wall - y_target
            dx = x_goal - x_spawn
            dy = y_virtual - y_spawn
        else:
            dx = x_goal - x_spawn
            dy = y_target - y_spawn

        # Force shooting toward negative x
        if dx >= -1e-6:
            dx = -0.1

        norm = math.sqrt(dx * dx + dy * dy)
        if norm < 1e-9:
            dx, dy, norm = -1.0, 0.0, 1.0

        dir_x = dx / norm
        dir_y = dy / norm

        speed = self.rng.uniform(self.speed_min, self.speed_max)
        v_local = [speed * dir_x, speed * dir_y, 0.0]

        self._spawn_ball_with_velocity_local(x_spawn, y_spawn, z_spawn, v_local)

    def _autoplay_maybe_spawn(self):
        if not self.autoplay:
            return
        if self.ball_id is not None:
            return
        if time.time() < self._next_shot_time:
            return
        self.reset_robot()
        self._spawn_and_shoot_random()

    def _autoplay_handle_terminal(self):
        if not self.autoplay or self.ball_id is None:
            return
        if not (self._goal_printed_for_ball or self._block_printed_for_ball):
            return

        try:
            p.removeBody(self.ball_id)
        except Exception:
            pass
        self.ball_id = None
        self._next_shot_time = time.time() + self.respawn_delay_s

    # ----------------------------
    # Goal / block detection
    # ----------------------------

    def _check_goal_rectangle_crossing(self):
        if self.ball_id is None:
            self._prev_ball_local_pos = None
            return

        bpos_w, born_w = p.getBasePositionAndOrientation(self.ball_id)
        lpos, _ = self._world_to_table_local(bpos_w, born_w)
        x1, y1, z1 = float(lpos[0]), float(lpos[1]), float(lpos[2])

        if self._prev_ball_local_pos is None:
            self._prev_ball_local_pos = (x1, y1, z1)
            return

        if self._goal_printed_for_ball:
            self._prev_ball_local_pos = (x1, y1, z1)
            return

        x0, y0, z0 = self._prev_ball_local_pos
        x_goal = self.goal_rect_x

        if not (x0 > x_goal and x1 <= x_goal):
            self._prev_ball_local_pos = (x1, y1, z1)
            return

        dx = x1 - x0
        if abs(dx) < 1e-12:
            self._prev_ball_local_pos = (x1, y1, z1)
            return

        t = (x_goal - x0) / dx
        if t < 0.0 or t > 1.0:
            self._prev_ball_local_pos = (x1, y1, z1)
            return

        y_hit = y0 + t * (y1 - y0)
        z_hit = z0 + t * (z1 - z0)

        y_min = self.goal_rect_y_min + self.ball_radius
        y_max = self.goal_rect_y_max - self.ball_radius
        z_min = self.goal_rect_z_min + self.ball_radius
        z_max = self.goal_rect_z_max - self.ball_radius

        if (y_min <= y_hit <= y_max) and (z_min <= z_hit <= z_max):
            print("GOAL")
            self._goal_printed_for_ball = True

        self._prev_ball_local_pos = (x1, y1, z1)

    def _check_block_event(self):
        if self.ball_id is None:
            self._stopped_steps = 0
            return

        if self._goal_printed_for_ball or self._block_printed_for_ball:
            return

        lin_w, _ = p.getBaseVelocity(self.ball_id)
        lin_l = self._world_vec_to_table_local(lin_w)
        vx = float(lin_l[0])
        speed = math.sqrt(lin_l[0] ** 2 + lin_l[1] ** 2 + lin_l[2] ** 2)

        if speed < self.block_speed_eps and abs(vx) < self.block_vx_eps:
            self._stopped_steps += 1
        else:
            self._stopped_steps = 0

        if self._stopped_steps >= self.block_stop_required_steps:
            print("BLOCK")
            self._block_printed_for_ball = True
            return

        if self._initial_vx_local is None or abs(self._initial_vx_local) < self.block_dir_eps:
            return
        if self.ball_steps < self.block_min_steps_before_reverse:
            return

        v0x = float(self._initial_vx_local)
        reversed_dir = (v0x < 0.0 and vx > self.block_reverse_vx_eps) or (v0x > 0.0 and vx < -self.block_reverse_vx_eps)
        if reversed_dir:
            print("BLOCK")
            self._block_printed_for_ball = True

    # ----------------------------
    # Step / run
    # ----------------------------

    def step(self):
        self._autoplay_maybe_spawn()

        p.stepSimulation()
        if self.ball_id is not None:
            self.ball_steps += 1

        self._check_goal_rectangle_crossing()
        self._check_block_event()
        self._autoplay_handle_terminal()

        if self.use_gui:
            self._update_ball_local_hud()
            time.sleep(self.dt)

    def close(self):
        p.disconnect(self.client)

    # ----------------------------
    # Camera
    # ----------------------------

    def _apply_camera(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=self.cam_dist,
            cameraYaw=self.cam_yaw,
            cameraPitch=self.cam_pitch,
            cameraTargetPosition=self.cam_target,
        )


def run_demo():
    sim = FoosballSim(use_gui=True, autoplay=True)
    while True:
        sim.step()


if __name__ == "__main__":
    run_demo()
