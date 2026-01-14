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
        add_wall_catchers: bool = True,
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

        self.client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.resetSimulation()
        p.setTimeStep(self.dt)
        p.setGravity(0, 0, -9.81)

        # Physics engine parameters (substeps + CCD friendliness)
        base_kwargs = dict(enableConeFriction=1, numSolverIterations=150)
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
        self.slider_idx, self.kicker_idx = self._get_joint_indices()
        self.slider_limits = self._get_limits(self.slider_idx)
        self.kicker_limits = self._get_limits(self.kicker_idx)
        self._disable_default_motors()

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

        # Per-shot bookkeeping
        self.ball_steps = 0
        self.initial_vx_local: Optional[float] = None

        # For goal detection crossing
        self._prev_ball_local_pos: Optional[Tuple[float, float, float]] = None

        # For block detection
        self._stopped_steps = 0

        # Optional catcher walls
        self.wall_catcher_ids: List[int] = []
        if self.add_wall_catchers:
            self.wall_catcher_ids = self._add_boundary_wall_catchers()
            self.env_body_ids.extend(self.wall_catcher_ids)
            self._disable_robot_vs_env_collisions(self.wall_catcher_ids)

        # HUD: scoreboard anchored above the goal mouth (GUI only)
        self._score_text_id: Optional[int] = None
        self._score_anchor_world = self.table_local_to_world_pos(
            [self.goal_rect_x, 0.0, self.goal_rect_z_max + 0.14]
        )
        # Debug overlays (GUI only)
        self._dbg_ball_id: Optional[int] = None
        self._dbg_player_id: Optional[int] = None
        self._dbg_aim_line_id: Optional[int] = None
        self._dbg_goal_point_id: Optional[int] = None
        self._est_pos_vis: Optional[np.ndarray] = None
        self._est_vel_vis: Optional[np.ndarray] = None

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
        # base_orn is identity in your setup; keep general for later
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

    def set_goal_intercept_debug(self, y_hit: Optional[float], z_hit: Optional[float]) -> None:
        """
        Draws a small marker on the goal plane at the predicted intercept (GUI only).
        Pass None to clear.
        """
        if not self.use_gui:
            return

        if y_hit is None or z_hit is None:
            self._remove_debug_item("_dbg_goal_point_id")
            return

        base_w = self.table_local_to_world_pos([self.goal_rect_x, y_hit, z_hit])
        top_w = self.table_local_to_world_pos([self.goal_rect_x, y_hit, z_hit + 0.03])
        self._dbg_goal_point_id = p.addUserDebugLine(
            base_w,
            top_w,
            lineColorRGB=(0.2, 1.0, 0.2),
            lineWidth=6,
            replaceItemUniqueId=self._dbg_goal_point_id if self._dbg_goal_point_id is not None else -1,
        )

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
    ) -> Dict[str, float]:
        """
        Spawns ball at x=0 or positive-x half, random y inside bounds.
        Launches toward goal plane x=goal_rect_x with y target in goal opening.
        Returns dict with the chosen parameters (for debugging).
        """
        self.remove_ball()

        margin = self.ball_radius + 0.02
        x_min, y_min, _ = self.table_min_local
        x_max, y_max, _ = self.table_max_local

        # y spawn: ensure entire ball fits
        y_spawn = self.rng.uniform(y_min + margin, y_max - margin)

        # x spawn: either 0 or positive half, but always safe
        x_goal = float(self.goal_rect_x)
        x_safe_min = max(x_goal + 0.06, x_min + margin)
        x_safe_max = x_max - margin
        if x_safe_max < x_safe_min:
            x_safe_max = x_safe_min

        if self.rng.random() < 0.5 and (x_safe_min <= 0.0 <= x_safe_max):
            x_spawn = 0.0
        else:
            lo = max(0.05, x_safe_min)
            x_spawn = self.rng.uniform(lo, x_safe_max) if lo <= x_safe_max else self.rng.uniform(x_safe_min, x_safe_max)

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

        if dx >= -1e-6:
            dx = -0.1

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

        # init block counter
        self._stopped_steps = 0

        return {
            "x_spawn": float(x_spawn),
            "y_spawn": float(y_spawn),
            "y_target": float(y_target),
            "speed": float(speed),
            "use_bounce": float(use_bounce),
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

    # ----------------------------
    # Player state helpers
    # ----------------------------

    def get_joint_positions(self) -> Tuple[float, float]:
        slider_pos = float(p.getJointState(self.robot_uid, self.slider_idx)[0])
        kicker_pos = float(p.getJointState(self.robot_uid, self.kicker_idx)[0])
        return slider_pos, kicker_pos

    def get_player_center_local(self) -> np.ndarray:
        # link index kicker_idx corresponds to the child link of joint "kicker"
        link_state = p.getLinkState(self.robot_uid, self.kicker_idx, computeForwardKinematics=True)
        pos_w = link_state[0]
        return self.world_to_table_local_pos(pos_w)

    def get_player_y_halfwidth(self) -> float:
        aabb_min, aabb_max = p.getAABB(self.robot_uid, self.kicker_idx)
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
    # Action application
    # ----------------------------

    def apply_action_targets(self, slider_target: float, kicker_target: float):
        p.setJointMotorControl2(
            self.robot_uid,
            self.slider_idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=float(slider_target),
            force=self.slider_limits.effort,
            maxVelocity=self.slider_limits.velocity,
        )
        p.setJointMotorControl2(
            self.robot_uid,
            self.kicker_idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=float(kicker_target),
            force=self.kicker_limits.effort,
            maxVelocity=self.kicker_limits.velocity,
        )

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
        """GOAL if ball crosses plane x=goal_rect_x and intersection is inside y/z rectangle."""
        if self.ball_id is None or self._prev_ball_local_pos is None:
            return False

        pos_l, _ = self.get_ball_true_local_pos_vel()
        x1, y1, z1 = float(pos_l[0]), float(pos_l[1]), float(pos_l[2])

        x0, y0, z0 = self._prev_ball_local_pos
        x_goal = float(self.goal_rect_x)

        self._prev_ball_local_pos = (x1, y1, z1)

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
        """
        BLOCK if (near defense zone) and (stops or reverses x) OR if contact with player link happens.
        This avoids counting wall bounces as blocks.
        """
        if self.ball_id is None:
            self._stopped_steps = 0
            return False

        pos_l, vel_l = self.get_ball_true_local_pos_vel()
        x, y, z = float(pos_l[0]), float(pos_l[1]), float(pos_l[2])
        vx = float(vel_l[0])
        speed = float(np.linalg.norm(vel_l))

        # Defense zone near goal plane
        defense_zone = abs(x - float(self.goal_rect_x)) < 0.12

        # Contact with player link (kicker_idx)
        contacts = p.getContactPoints(bodyA=self.ball_id, bodyB=self.robot_uid, linkIndexB=self.kicker_idx)
        had_contact = (contacts is not None and len(contacts) > 0)

        # Stop condition (sustained)
        if defense_zone and speed < 0.08:
            self._stopped_steps += 1
        else:
            self._stopped_steps = 0

        if self._stopped_steps >= 10:
            return True

        # Reversal relative to initial vx (only in defense zone)
        if defense_zone and self.initial_vx_local is not None:
            v0x = float(self.initial_vx_local)
            if abs(v0x) > 0.05:
                reversed_dir = (v0x < 0.0 and vx > 0.10) or (v0x > 0.0 and vx < -0.10)
                if reversed_dir:
                    return True

        # Contact-based block (immediate)
        if defense_zone and had_contact:
            return True

        return False

    # ----------------------------
    # Simulation stepping
    # ----------------------------

    def step_sim(self, n_steps: int):
        for _ in range(int(n_steps)):
            p.stepSimulation()
            if self.ball_id is not None:
                self.ball_steps += 1
            # Debug overlays for GUI: player center, ball, aim line
            self._update_debug_markers()
