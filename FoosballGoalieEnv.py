from typing import Optional, Tuple, Dict, Any
from collections import deque
import time
import math

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from FoosballSimCore import _FoosballSimCore


class FoosballGoalieEnv(gym.Env):
    """
    Policy acts on BOTH position and velocity for BOTH joints.

    Action (4, in [-1,1]):
      [ slider_pos_cmd, slider_vel_cmd, kicker_pos_cmd, kicker_vel_cmd ]

    - slider_pos_cmd maps to slider joint limits
    - kicker_pos_cmd maps to kicker joint limits
    - slider_vel_cmd maps to [0, slider_vel_cap_mps]
    - kicker_vel_cmd maps to [0, kicker_vel_cap_rads]

    Policy runs at policy_hz (default 20 Hz).
    Simulation runs at sim_hz (default 1000 or 2000 Hz).
    """

    metadata = {"render_modes": ["human", "none"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str = "none",
        # control rates
        policy_hz: float = 20.0,
        sim_hz: int = 1000,  # set to 2000 if desired
        max_episode_steps: int = 100,
        seed: Optional[int] = None,
        num_substeps: int = 8,
        # shot params
        speed_min: float = 2.0,
        speed_max: float = 10.0,
        bounce_prob: float = 0.25,
        # camera noise
        cam_noise_std: Tuple[float, float, float] = (0.002, 0.002, 0.002),
        # actuator caps (what the policy can request)
        slider_vel_cap_mps: float = 20.0,
        kicker_vel_cap_rads: float = 170.0,
        # GUI pacing
        real_time_gui: bool = True,
    ):
        super().__init__()

        if render_mode not in ("human", "none"):
            raise ValueError("render_mode must be 'human' or 'none'")

        self.render_mode = render_mode

        self.policy_hz = float(policy_hz)
        self.sim_hz = int(sim_hz)
        if self.sim_hz <= 0:
            raise ValueError("sim_hz must be > 0")
        if self.policy_hz <= 0:
            raise ValueError("policy_hz must be > 0")

        self.dt_sim = 1.0 / float(self.sim_hz)
        self.dt_policy = 1.0 / float(self.policy_hz)

        # how many sim steps per policy step
        self.steps_per_policy = int(round(self.dt_policy / self.dt_sim))
        self.steps_per_policy = max(1, self.steps_per_policy)

        # keep an exact effective dt for prediction horizon
        self.dt_eff = float(self.steps_per_policy) * float(self.dt_sim)

        self.max_episode_steps = int(max_episode_steps)

        self.speed_min = float(speed_min)
        self.speed_max = float(speed_max)
        self.bounce_prob = float(bounce_prob)

        self.cam_noise_std = np.array(cam_noise_std, dtype=np.float32)

        self.slider_vel_cap_mps = float(slider_vel_cap_mps)
        self.kicker_vel_cap_rads = float(kicker_vel_cap_rads)

        self.real_time_gui = bool(real_time_gui)

        # Action now includes pos+vel for slider+kicker
        # [-1,1]^4
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # obs: 15 dims (est pos/vel/pred_pos + joints + intercept features)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

        # sim core
        self.sim = _FoosballSimCore(
            use_gui=(render_mode == "human"),
            time_step=self.dt_sim,   # <-- sim runs at 1000/2000 Hz
            seed=seed,
            ball_restitution=0.8,
            table_restitution=0.5,
            wall_restitution=0.8,
            add_wall_catchers=False,
            num_substeps=int(num_substeps),
        )

        # estimator state
        self._est_pos = np.zeros(3, dtype=np.float32)
        self._est_vel = np.zeros(3, dtype=np.float32)
        self._pred_pos = np.zeros(3, dtype=np.float32)
        self._vel_win = deque(maxlen=3)
        self._pos_hist = []  # list of (t, pos) for smoothing

        # episode bookkeeping
        self._episode_step = 0
        self._terminated_event: Optional[str] = None
        self._goal_count = 0
        self._block_count = 0
        self._out_count = 0
        self._intercept_available = False

        # HUD
        self._update_scoreboard(highlight=None)
        if self.render_mode == "human":
            self.sim.set_goal_intercept_debug(None, None)

    # ----------------------------
    # Gymnasium API
    # ----------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._episode_step = 0
        self._terminated_event = None

        self.sim.remove_ball()
        self.sim.reset_robot_randomized()

        shot_info = self.sim.spawn_shot_random(self.speed_min, self.speed_max, self.bounce_prob)

        self._update_estimator(first=True)
        self.sim.set_estimated_ball_state(self._est_pos, self._est_vel)

        obs = self._get_obs()
        info = {"shot": shot_info}
        self._update_scoreboard(highlight=None)
        return obs, info

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape != (4,):
            raise ValueError(f"Action must have shape (4,), got {a.shape}")

        self._episode_step += 1

        # unpack
        a_sp = float(np.clip(a[0], -1.0, 1.0))  # slider position command
        a_sv = float(np.clip(a[1], -1.0, 1.0))  # slider velocity command
        a_kp = float(np.clip(a[2], -1.0, 1.0))  # kicker position command
        a_kv = float(np.clip(a[3], -1.0, 1.0))  # kicker velocity command

        # map positions [-1,1] -> joint limits
        slider_target = (
            self.sim.slider_limits.lower
            + (a_sp + 1.0) * 0.5 * (self.sim.slider_limits.upper - self.sim.slider_limits.lower)
        )
        kicker_target = (
            self.sim.kicker_limits.lower
            + (a_kp + 1.0) * 0.5 * (self.sim.kicker_limits.upper - self.sim.kicker_limits.lower)
        )

        # map velocity commands [-1,1] -> [0, cap]
        # (If you prefer signed velocity intent, we can change this, but POSITION_CONTROL uses maxVelocity as a cap.)
        slider_vel_cap = (a_sv + 1.0) * 0.5 * self.slider_vel_cap_mps
        kicker_vel_cap = (a_kv + 1.0) * 0.5 * self.kicker_vel_cap_rads

        # apply once per policy step, then substep sim
        self.sim.apply_action_targets(
            slider_target,
            kicker_target,
            slider_vel_cap=float(slider_vel_cap),
            kicker_vel_cap=float(kicker_vel_cap),
        )

        terminated = False
        truncated = False
        event = None
        out_reason = ""

        for _ in range(self.steps_per_policy):
            self.sim.step_sim(1)

            # estimator at sim rate; dt_scale=1.0 means dt_sim
            self._update_estimator(first=False, dt_scale=1.0)
            self.sim.set_estimated_ball_state(self._est_pos, self._est_vel)

            # add debug for goal
            intercept = self._predict_goal_intercept_est()
            self._debug_goal_intercept(intercept)

            if self.render_mode == "human" and self.real_time_gui:
                time.sleep(self.dt_sim / 2)

            if self.sim.check_goal_crossing():
                terminated = True
                event = "goal"
                break
            if self.sim.check_block_event():
                terminated = True
                event = "block"
                break

            out, reason = self.sim.check_ball_out_of_bounds()
            if out:
                truncated = True
                event = "out"
                out_reason = reason
                break

        self._terminated_event = event
        reward = self._get_reward(event)

        if self._episode_step >= self.max_episode_steps:
            truncated = True

        obs = self._get_obs()
        info = self._get_info(event)
        if event == "out":
            info["out_reason"] = out_reason

        if event == "goal":
            self._goal_count += 1
        elif event == "block":
            self._block_count += 1
        elif event == "out":
            self._out_count += 1

        if event is not None:
            self._update_scoreboard(highlight=event)

        return obs, float(reward), bool(terminated), bool(truncated), info

    # ----------------------------
    # Observation / estimator
    # ----------------------------

    def _sample_noisy_ball_pos_local(self) -> np.ndarray:
        pos_true, _ = self.sim.get_ball_true_local_pos_vel()
        noise = np.array(
            [
                self.np_random.normal(0.0, float(self.cam_noise_std[0])),
                self.np_random.normal(0.0, float(self.cam_noise_std[1])),
                self.np_random.normal(0.0, float(self.cam_noise_std[2])),
            ],
            dtype=np.float32,
        )
        return pos_true.astype(np.float32) + noise

    def _update_estimator(self, first: bool, dt_scale: Optional[float] = None):
        pos = self._sample_noisy_ball_pos_local()

        # dt_eff for this estimator update
        if dt_scale is None:
            dt_eff = float(self.dt_eff)
        else:
            dt_eff = float(self.dt_sim) * float(dt_scale)

        if first:
            self._est_pos = pos.copy()
            self._est_vel[:] = 0.0
            self._pred_pos = pos.copy()
            self._vel_win.clear()
            self._vel_win.append((0.0, pos.copy()))
            self._pos_hist = [(0.0, pos.copy())]
            return

        t_prev = self._vel_win[-1][0] if len(self._vel_win) > 0 else 0.0
        t_new = t_prev + dt_eff
        noise_tol = float(max(self.cam_noise_std)) * 3.0  # a few mm by default
        speed_tol = 0.8  # m/s threshold to treat deviations as meaningful

        # Direction change or significant deviation -> drop past measurements.
        # Allow resets only after some travel/speed to ignore slow jitter.
        min_travel_for_deviation = 0.025  # meters
        min_travel_for_reset = 0.045      # meters
        if len(self._pos_hist) > 0:
            base_pos = self._pos_hist[0][1]
            last_pos = self._pos_hist[-1][1]
            main_vec = last_pos - base_pos
            step_vec = pos - last_pos
            main_len = float(np.linalg.norm(main_vec))
            step_len = float(np.linalg.norm(step_vec))

            # Detect a wall bounce from the latest segment alone (robust against long history).
            y_min_wall = float(self.sim.table_min_local[1] + self.sim.ball_radius + 0.001)
            y_max_wall = float(self.sim.table_max_local[1] - self.sim.ball_radius - 0.001)
            y0 = float(last_pos[1])
            y1 = float(pos[1])
            bounce_wall_y: Optional[float] = None
            if (y0 < y_min_wall and y1 >= y_min_wall) or (y0 > y_min_wall and y1 <= y_min_wall):
                bounce_wall_y = y_min_wall
            if (y0 < y_max_wall and y1 >= y_max_wall) or (y0 > y_max_wall and y1 <= y_max_wall):
                # If both bounds are crossed (numerical glitch), prefer the first hit along the segment.
                if bounce_wall_y is None:
                    bounce_wall_y = y_max_wall

            if bounce_wall_y is not None and abs(float(step_vec[1])) > 1e-6:
                # Rebase history at the rebound point so the line fit uses only the post-bounce path.
                t_hit = float((bounce_wall_y - y0) / float(step_vec[1]))
                t_hit = max(0.0, min(1.0, t_hit))
                bounce_pos = last_pos + step_vec * t_hit
                dt_post = max(dt_eff * (1.0 - t_hit), 1e-6)
                self._vel_win.clear()
                self._pos_hist = [(0.0, bounce_pos.astype(np.float32))]
                self._vel_win.append((0.0, bounce_pos.astype(np.float32)))
                self._pos_hist.append((dt_post, pos.copy()))
                vel_post = (pos - bounce_pos) / dt_post
                self._est_pos = pos.copy()
                self._est_vel = vel_post.astype(np.float32)
                self._pred_pos = (self._est_pos + self._est_vel * float(self.dt_eff)).astype(np.float32)
                self.sim.set_estimated_ball_state(self._est_pos, self._est_vel)
                return

            # Explicit wall-bounce detection on X/Y (ignore Z since we stay on the surface).
            if len(self._pos_hist) >= 2:
                prev_pos = self._pos_hist[-2][1]
                dx_prev = float(last_pos[0] - prev_pos[0])
                dy_prev = float(last_pos[1] - prev_pos[1])
                dx_now = float(step_vec[0])
                dy_now = float(step_vec[1])

                x_min = float(self.sim.table_min_local[0] + self.sim.ball_radius + 0.002)
                x_max = float(self.sim.table_max_local[0] - self.sim.ball_radius - 0.002)
                y_min = float(self.sim.table_min_local[1] + self.sim.ball_radius + 0.002)
                y_max = float(self.sim.table_max_local[1] - self.sim.ball_radius - 0.002)

                near_x_wall = (abs(float(last_pos[0]) - x_min) < 0.01) or (abs(float(last_pos[0]) - x_max) < 0.01)
                near_y_wall = (abs(float(last_pos[1]) - y_min) < 0.01) or (abs(float(last_pos[1]) - y_max) < 0.01)

                bounced_x = near_x_wall and (dx_prev * dx_now < -1e-6) and abs(dx_prev) > 1e-4 and abs(dx_now) > 1e-4
                bounced_y = near_y_wall and (dy_prev * dy_now < -1e-6) and abs(dy_prev) > 1e-4 and abs(dy_now) > 1e-4

                if bounced_x or bounced_y:
                    axis = "x" if bounced_x else "y"
                    # print(
                    #     f"[Estimator] reset: wall bounce on {axis}, "
                    #     f"dx_prev={dx_prev:.4f}, dx_now={dx_now:.4f}, "
                    #     f"dy_prev={dy_prev:.4f}, dy_now={dy_now:.4f}, "
                    #     f"pos=({last_pos[0]:.4f},{last_pos[1]:.4f})"
                    # )
                    self._vel_win.clear()
                    self._pos_hist = [(0.0, pos.copy())]
                    self._est_pos = pos.copy()
                    self._est_vel[:] = 0.0
                    self._pred_pos = pos.copy()
                    self.sim.set_estimated_ball_state(self._est_pos, self._est_vel)
                    return

            dir_flip = False
            if main_len > 1e-6 and step_len > 1e-6:
                dir_flip = float(np.dot(step_vec, main_vec)) < -1e-4

            line_deviation = False
            dev = 0.0
            travel = float(np.linalg.norm(last_pos - base_pos))
            if travel >= min_travel_for_deviation and main_len > 1e-6:
                # Distance of new point to line defined by (base_pos, last_pos).
                rel = pos - base_pos
                proj = float(np.dot(rel, main_vec)) / (main_len * main_len)
                closest = base_pos + proj * main_vec
                dev = float(np.linalg.norm(pos - closest))
                # Require deviation well outside noise band to avoid over-reset.
                line_deviation = dev > max(noise_tol * 2.0, noise_tol + 0.003)

            hist_dt = t_new - self._pos_hist[0][0]
            avg_speed = travel / max(hist_dt, 1e-6)
            step_speed = step_len / max(dt_eff, 1e-6)
            allow_reset = (avg_speed >= speed_tol and travel >= min_travel_for_reset)

            if allow_reset and len(self._pos_hist) >= 3 and (dir_flip or line_deviation):
                # print(
                #     f"[Estimator] reset: dir_flip={dir_flip}, line_dev={line_deviation}, "
                #     f"step_speed={step_speed:.3f} m/s, avg_speed={avg_speed:.3f} m/s, "
                #     f"travel={travel:.3f} m, dev={dev:.4f}, dev_tol={noise_tol:.4f}"
                # )
                self._vel_win.clear()
                self._pos_hist = [(0.0, pos.copy())]
                self._est_pos = pos.copy()
                self._est_vel[:] = 0.0
                self._pred_pos = pos.copy()
                self.sim.set_estimated_ball_state(self._est_pos, self._est_vel)
                return

        self._vel_win.append((t_new, pos.copy()))
        self._pos_hist.append((t_new, pos.copy()))

        vel = np.zeros(3, dtype=np.float32)
        if len(self._vel_win) >= 3:
            (t0, p0) = self._vel_win[0]
            (t2, p2) = self._vel_win[-1]
            denom = float(t2 - t0)
            if denom > 1e-9:
                vel = (p2 - p0) / denom
        elif len(self._vel_win) == 2:
            (t0, p0) = self._vel_win[0]
            (t1, p1) = self._vel_win[1]
            denom = float(t1 - t0)
            if denom > 1e-9:
                vel = (p1 - p0) / denom

        # Estimate position/velocity with a single linear fit and evaluate it at the latest timestamp.
        if len(self._pos_hist) >= 2:
            times = np.array([t for t, _ in self._pos_hist], dtype=np.float32)
            coords = np.stack([p for _, p in self._pos_hist], axis=0)
            t_norm = times - times[0]
            t_mean = float(np.mean(t_norm))
            p_mean = np.mean(coords, axis=0)
            denom = float(np.sum((t_norm - t_mean) ** 2))
            if denom > 1e-8:
                slopes = np.sum((t_norm - t_mean).reshape(-1, 1) * (coords - p_mean), axis=0) / denom
                intercepts = p_mean - slopes * t_mean
                t_curr = float(t_norm[-1])
                self._est_vel = slopes.astype(np.float32)
                self._est_pos = (intercepts + slopes * t_curr).astype(np.float32)
            else:
                self._est_vel = vel.astype(np.float32)
                self._est_pos = pos.copy()
        else:
            self._est_vel = vel.astype(np.float32)
            self._est_pos = pos.copy()

        # predict one policy-step ahead (not one sim step)
        self._pred_pos = (self._est_pos + self._est_vel * float(self.dt_eff)).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        slider_pos, kicker_pos = self.sim.get_joint_positions()
        intercept = self._predict_goal_intercept_est()
        if intercept is None:
            y_hit = 0.0
            z_hit = 0.0
            t_hit = 0.0
            intercept_flag = 0.0
        else:
            y_hit, z_hit, _x_plane, t_hit = intercept
            intercept_flag = 1.0

        obs = np.concatenate(
            [
                self._est_pos.astype(np.float32),
                self._est_vel.astype(np.float32),
                self._pred_pos.astype(np.float32),
                np.array([kicker_pos, slider_pos], dtype=np.float32),
                np.array([y_hit, z_hit, t_hit, intercept_flag], dtype=np.float32),
            ],
            axis=0,
        )
        return obs

    # ----------------------------
    # Reward / info (unchanged)
    # ----------------------------

    def _predict_goal_intercept_est(self) -> Optional[Tuple[float, float, float, float]]:
        if self.sim.ball_id is None:
            return None

        x, y, z = map(float, self._est_pos)
        vx, vy, vz = map(float, self._est_vel)

        if vx >= -1e-4:
            return None

        # Use goalie plane while the ball is in front of the goalie; once it passes, switch to the goal line.
        # This keeps the predicted intercept consistent with the reward box jump.
        if x <= float(self.sim.goalie_x) - float(self.sim.ball_radius):
            x_goal = float(self.sim.goal_rect_x)
        else:
            x_goal = float(self.sim.goalie_x)
        y_min = float(self.sim.table_min_local[1] + self.sim.ball_radius)
        y_max = float(self.sim.table_max_local[1] - self.sim.ball_radius)

        t_accum = 0.0
        for _ in range(2):
            t_goal = (x_goal - x) / vx
            if t_goal <= 0.0:
                return None

            t_wall = float("inf")
            if abs(vy) > 1e-6:
                t_wall = (y_max - y) / vy if vy > 0 else (y_min - y) / vy

            if 0.0 < t_wall < t_goal:
                x += vx * t_wall
                y = y_max if vy > 0 else y_min
                z += vz * t_wall
                vy = -vy
                t_accum += t_wall
                continue

            y_hit = y + vy * t_goal
            z_hit = z + vz * t_goal
            t_accum += t_goal
            return (y_hit, z_hit, x_goal, t_accum)

        return None

    def _dense_alignment_reward(self) -> float:
        intercept = self._predict_goal_intercept_est()
        self._debug_goal_intercept(intercept)
        if intercept is None:
            self._intercept_available = False
            return 0.0

        self._intercept_available = True
        y_pred, z_pred, x_plane, _t_hit = intercept

        # Flatten path in Z: keep current estimated height and ignore vertical velocity.
        z_flat = float(self._est_pos[2])

        vx, vy, vz = map(float, self._est_vel)
        vz = 0.0
        if vx >= -1e-4:
            return 0.0

        # Compute path segment from goalie plane to goal plane using estimated velocity.
        x_goalie = float(self.sim.goalie_x)
        x_goal = float(self.sim.goal_rect_x)
        x0, y0, z0 = map(float, self._est_pos)
        z0 = z_flat

        # Parametric times to planes.
        t_goalie = (x_goalie - x0) / vx
        t_goal_line = (x_goal - x0) / vx
        if t_goalie <= 0.0:
            return 0.0

        # Ensure goal plane time is after goalie plane along travel.
        if t_goal_line <= t_goalie:
            t_goal_line = t_goalie + 0.05

        start = np.array([x0 + vx * t_goalie, y0 + vy * t_goalie, z0 + vz * t_goalie], dtype=np.float32)
        end = np.array([x0 + vx * t_goal_line, y0 + vy * t_goal_line, z0 + vz * t_goal_line], dtype=np.float32)
        seg_dir = end - start
        seg_len = float(np.linalg.norm(seg_dir))
        if seg_len < 1e-6:
            return 0.0
        seg_dir /= seg_len

        toe_center, toe_half = self.sim._get_player_toebox(home=True)
        toe_center = toe_center.astype(np.float32)
        toe_half = np.asarray(toe_half, dtype=np.float32)

        # Closest point on segment to toe.
        rel = toe_center - start
        t_proj = float(np.dot(rel, seg_dir))
        t_clamped = max(0.0, min(seg_len, t_proj))
        closest = start + seg_dir * t_clamped
        dist = float(np.linalg.norm(toe_center - closest))
        # Account for toe box size by shrinking the distance.
        toe_radius = float(np.linalg.norm(toe_half))  # spherical approximation of toe extents
        dist = max(0.0, dist)

        # Use the same y/z tolerances as before to scale decay.
        player_y_half = float(self.sim.get_player_y_halfwidth())
        denom = max(player_y_half + float(self.sim.ball_radius), 1e-6)
        z_span = float(self.sim.goal_rect_z_max - self.sim.goal_rect_z_min)
        z_tol = 0.5 * z_span + 0.05
        tol = max(denom, z_tol, toe_radius, 1.0)

        # Distance along path (meters).
        d = dist
        close_cutoff = 0.05  # 5 cm

        # Fine reward: only inside 5 cm, exponential shaped to be 1.0 at d=0 and 0.0 at d=close_cutoff.
        fine_decay = max(close_cutoff * 0.4, 1e-3)  # steeper inside the close band
        base = math.exp(-close_cutoff / fine_decay)
        fine_raw = math.exp(-d / fine_decay)
        fine = 0.0
        if d <= close_cutoff:
            fine = float(max(0.0, (fine_raw - base) / max(1.0 - base, 1e-6)))
        # Cap fine at 1.0 by construction.

        # Coarse reward: exponential toward 0.5, peaked once you are within 5 cm (flat-top after that).
        coarse_decay = max(close_cutoff * 0.6, 1e-3)
        if d <= close_cutoff:
            coarse = 0.1
        else:
            coarse = float(0.1 * math.exp(-(d - close_cutoff) / coarse_decay))

        return fine + coarse

    def _debug_goal_intercept(self, intercept: Optional[Tuple[float, float, float, float]]) -> None:
        if self.render_mode != "human":
            return
        if intercept is None:
            self.sim.set_goal_intercept_debug(None, None)
            return
        y_pred, z_pred, x_plane, _t_hit = intercept
        # Flatten the visual path onto the XY plane (ignore vertical component).
        v_dir = self._est_vel.astype(np.float32)
        v_dir[2] = 0.0
        self.sim.set_goal_intercept_debug(float(y_pred), float(z_pred), float(x_plane), tuple(v_dir))
        # print(f"Predicted intercept at y={y_pred:.3f}, z={z_pred:.3f}")

    def _get_reward(self, event: Optional[str]) -> float:
        dense = 0.0 # self._dense_alignment_reward()
        sparse = 0.0
        if event == "block":
            sparse += 10.0
        elif event == "goal":
            sparse -= 10.0
        elif event == "out":
            sparse -= 2.0
        return float(dense + sparse)

    def _get_info(self, event: Optional[str]) -> Dict[str, Any]:
        pos_true, vel_true = self.sim.get_ball_true_local_pos_vel()
        player_c = self.sim.get_player_center_local()
        slider_pos, kicker_pos = self.sim.get_joint_positions()

        (spos, kpos), (svel, kvel) = self.sim.get_joint_positions_and_vels()
        intercept = self._predict_goal_intercept_est()
        if intercept is None:
            intercept_dict = None
        else:
            y_hit, z_hit, x_plane, t_hit = intercept
            intercept_dict = {
                "y": float(y_hit),
                "z": float(z_hit),
                "x_plane": float(x_plane),
                "t_hit": float(t_hit),
            }

        return {
            "event": event,
            "dense": float(self._dense_alignment_reward()),
            "ball_true_pos": pos_true.astype(np.float32),
            "ball_true_vel": vel_true.astype(np.float32),
            "ball_est_pos": self._est_pos.astype(np.float32),
            "ball_est_vel": self._est_vel.astype(np.float32),
            "player_center": player_c.astype(np.float32),
            "slider_pos": float(slider_pos),
            "kicker_pos": float(kicker_pos),
            "slider_vel": float(svel),
            "kicker_vel": float(kvel),
            "intercept_available": bool(self._intercept_available),
            "intercept_pred": intercept_dict,
            "dt_sim": float(self.dt_sim),
            "dt_policy": float(self.dt_policy),
            "steps_per_policy": int(self.steps_per_policy),
        }

    def _update_scoreboard(self, highlight: Optional[str]):
        if hasattr(self, "sim"):
            try:
                self.sim.update_scoreboard_text(self._goal_count, self._block_count, self._out_count, highlight)
            except AttributeError:
                pass


if __name__ == "__main__":
    # Smoke test: random actions (now 4D)
    env = FoosballGoalieEnv(render_mode="human", seed=0, policy_hz=20.0, sim_hz=1000, real_time_gui=True)
    obs, info = env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
