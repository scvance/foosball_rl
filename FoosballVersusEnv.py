from typing import Optional, Tuple, Dict, Any
from collections import deque
import time
import math

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from FoosballSimCore import _FoosballSimCore


class FoosballVersusEnv(gym.Env):
    """
    Two-goalie self-play environment.
    - Action: Dict with {"home": [slider, kicker], "away": [slider, kicker]} in [-1, 1]
    - Observation per side: ball est pos/vel/pred (9) + own joints (2) + opp joints (2) => 13 dims
    - Reward: basic zero-sum placeholder (you can swap in your own later).
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
        serve_side: str = "random",  # 'home' | 'away' | 'random'
    ):
        super().__init__()

        if render_mode not in ("human", "none"):
            raise ValueError("render_mode must be 'human' or 'none'")
        
        self.serve_side = serve_side

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
        act_box = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Dict({"home": act_box, "away": act_box})
        # obs: 15 dims (est pos/vel/pred_pos + joints + intercept features)
        obs_box = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.observation_space = spaces.Dict({"home": obs_box, "away": obs_box})

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
        if self.render_mode == "human":
            self.sim.set_goal_intercept_debug(None, None, None, None)

        # Ensure opponent joints exist (URDF must define opponent_slider/opponent_kicker).
        if self.sim.opponent_slider_idx is None or self.sim.opponent_kicker_idx is None:
            raise RuntimeError(
                "FoosballVersusEnv expects URDF joints 'opponent_slider' and 'opponent_kicker'. "
                "Update the URDF before using this env."
            )

    # ----------------------------
    # Gymnasium API
    # ----------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._episode_step = 0
        self._terminated_event = None

        # reset sim
        self.sim.remove_ball()
        self.sim.reset_robot_randomized(self.np_random)

        # spawn a new shot (side configurable for self-play)
        shot_side = self.serve_side
        if shot_side == "random":
            shot_side = "home" if self.np_random.random() < 0.5 else "away"
        shot_info = self.sim.spawn_shot_random(self.speed_min, self.speed_max, self.bounce_prob, target=shot_side)

        # init estimator from noisy position
        self._update_estimator(first=True)
        self.sim.set_estimated_ball_state(self._est_pos, self._est_vel)

        obs = self._get_obs()
        info = {"shot": shot_info}
        return obs, info

    def step(self, action: Dict[str, np.ndarray]):
        a_home = np.asarray(action["home"], dtype=np.float32).reshape(-1)
        a_away = np.asarray(action["away"], dtype=np.float32).reshape(-1)
        if a_home.shape != (4,) or a_away.shape != (4,):
            raise ValueError(f"Actions must have shape (4,) each, got {a_home.shape} and {a_away.shape}")

        self._episode_step += 1

        # map action [-1,1] to joint ranges
        home_slider_target = self._interp_action(a_home[0], self.sim.slider_limits)
        home_slider_vel = self._interp_velocity(a_home[1], self.sim.slider_limits)
        home_kicker_target = self._interp_action(a_home[2], self.sim.kicker_limits)
        home_kicker_vel = self._interp_velocity(a_home[3], self.sim.kicker_limits)

        opp_slider_limits = self.sim.opponent_slider_limits
        opp_kicker_limits = self.sim.opponent_kicker_limits
        away_slider_target = self._interp_action(a_away[0], opp_slider_limits)
        away_slider_vel = self._interp_velocity(a_away[1], opp_slider_limits)
        away_kicker_target = self._interp_action(a_away[2], opp_kicker_limits)
        away_kicker_vel = self._interp_velocity(a_away[3], opp_kicker_limits)

        self.sim.apply_action_targets_dual(
            home_slider_target,
            home_kicker_target,
            away_slider_target,
            away_kicker_target,
            home_slider_vel_cap=home_slider_vel,
            home_kicker_vel_cap=home_kicker_vel,
            away_slider_vel_cap=away_slider_vel,
            away_kicker_vel_cap=away_kicker_vel,
        )

        terminated = False
        truncated = False
        event = None
        out_reason = ""
        last_block_event = None

        last_contacts = {"home": False, "away": False}
        for _ in range(self.steps_per_policy):
            self.sim.step_sim(1)
            y_pred, z_pred, x_goal, t_goal = self._predict_goal_intercept_est("away") or (0.0, 0.0, 0.0, 0.0)
            self.sim.set_goal_intercept_debug(y_pred, z_pred, x_goal, v_dir=(1.0, 0.0, 0.0))

            # Update estimator every sim step using per-step dt for smoother debug markers
            self._update_estimator(first=False, dt_scale=1.0)
            self.sim.set_estimated_ball_state(self._est_pos, self._est_vel)

            if self.render_mode == "human":
                time.sleep(self.dt_sim / 2) if self.real_time_gui else 0.0

            goals = self.sim.check_goal_crossings_dual()
            blocks = self.sim.check_block_events_dual()
            last_contacts["home"] = last_contacts["home"] or self.sim.had_contact_with_kicker(self.sim.kicker_idx)
            last_contacts["away"] = last_contacts["away"] or self.sim.had_contact_with_kicker(self.sim.opponent_kicker_idx)
            out, reason = self.sim.check_ball_out_of_bounds()

            if goals.get("home", False):
                terminated = True
                event = "home_goal"
                break
            if goals.get("away", False):
                terminated = True
                event = "away_goal"
                break
            if blocks.get("home", False):
                last_block_event = "home_block"
            if blocks.get("away", False):
                last_block_event = "away_block"
            if out:
                truncated = True
                event = "out"
                out_reason = reason
                break

        if event is None and last_block_event is not None:
            event = last_block_event

        self._terminated_event = event
        rewards = self._get_rewards(event, last_contacts)

        if self._episode_step >= self.max_episode_steps:
            truncated = True

        obs = self._get_obs()
        info = self._get_info(event)
        if event == "out":
            info["out_reason"] = out_reason

        return obs, rewards, bool(terminated), bool(truncated), info

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

    def _get_obs_for_side(self, side: str) -> np.ndarray:
        (slider_home, kicker_home), (v_slider_home, v_kicker_home) = self.sim.get_joint_positions_and_vels()
        (slider_away, kicker_away), (v_slider_away, v_kicker_away) = self.sim.get_opponent_joint_positions_and_vels()

        if side == "home":
            own_joints = np.array([kicker_home, slider_home, v_kicker_home, v_slider_home], dtype=np.float32)
            opp_joints = np.array(
                [
                    kicker_away if kicker_away is not None else 0.0,
                    slider_away if slider_away is not None else 0.0,
                    v_kicker_away if v_kicker_away is not None else 0.0,
                    v_slider_away if v_slider_away is not None else 0.0,
                ],
                dtype=np.float32,
            )
        else:
            own_joints = np.array(
                [
                    kicker_away if kicker_away is not None else 0.0,
                    slider_away if slider_away is not None else 0.0,
                    v_kicker_away if v_kicker_away is not None else 0.0,
                    v_slider_away if v_slider_away is not None else 0.0,
                ],
                dtype=np.float32,
            )
            opp_joints = np.array([kicker_home, slider_home, v_kicker_home, v_slider_home], dtype=np.float32)

        y_pred, z_pred, x_goal, t_goal = self._predict_goal_intercept_est(side) or (0.0, 0.0, 0.0, 0.0)

        obs = np.concatenate(
            [
                self._est_pos.astype(np.float32),  # 3
                self._est_vel.astype(np.float32),  # 3
                self._pred_pos.astype(np.float32),  # 3
                own_joints,  # 4
                opp_joints,  # 4
                np.array([y_pred, z_pred, x_goal, t_goal], dtype=np.float32)
            ],
            axis=0,
        )
        return obs

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {
            "home": self._get_obs_for_side("home"),
            "away": self._get_obs_for_side("away"),
        }

    # ----------------------------
    # Reward
    # ----------------------------

    def _predict_goal_intercept_est(self, side) -> Optional[Tuple[float, float, float, float]]:
        if self.sim.ball_id is None:
            return None

        x, y, z = map(float, self._est_pos)
        vx, vy, vz = map(float, self._est_vel)
        # Only predict if ball is moving toward the goal.
        if vx >= -1e-4 and side == "home":
            return None
        
        if vx <= 1e-4 and side == "away":
            return None
        
        if side == "home":
            goal_rect_x = float(self.sim.goal_rect_x)
            goalie_x = float(self.sim.goalie_x)
        else:
            goal_rect_x = float(self.sim.goal_rect_x_away)
            goalie_x = float(self.sim.goalie_x_away)

        # Use goalie plane while the ball is in front of the goalie; once it passes, switch to the goal line.
        # This keeps the predicted intercept consistent with the reward box jump.
        if x <= goalie_x - float(self.sim.ball_radius):
            x_goal = goal_rect_x
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

    def _dense_alignment_reward(self, side: str) -> float:
        """
        Dense 0..1 reward for being aligned with the predicted intercept line at the player plane.
        Only paid if ball is moving toward your goal.
        """
        vx = float(self._est_vel[0])
        if side == "home" and vx >= -1e-4:
            return 0.0
        if side == "away" and vx <= 1e-4:
            return 0.0

        intercept = self._predict_goal_intercept_est(side)
        if intercept is None:
            return 0.0

        y_pred, z_pred, _, _ = intercept

        # player position / coverage
        if side == "home":
            player_c = self.sim.get_player_center_local()
            y_half = self.sim.get_player_y_halfwidth()
        else:
            player_c = self.sim.get_opponent_player_center_local()
            y_half = self.sim.get_opponent_y_halfwidth()
            if player_c is None or y_half is None:
                return 0.0

        y_p = float(player_c[1])

        denom = y_half + float(self.sim.ball_radius)

        y_err = abs(y_pred - y_p)
        r_close = max(0.0, 1.0 - (y_err / max(denom, 1e-6)))
        r_tight = max(0.0, 1.0 - (y_err / max(0.4 * denom, 1e-6)))

        # optional: gate by z being inside approximate mouth height
        z_ok = (self.sim.goal_rect_z_min - 0.02) <= z_pred <= (self.sim.goal_rect_z_max + 0.05)

        if not z_ok:
            return 0.0

        # Blend close and "right on" tiers
        return float(0.5 * r_close + 0.5 * r_tight)

    def _kick_velocity_reward(self, side: str, had_contact: bool) -> float:
        if not had_contact:
            return 0.0
        _, vel_true = self.sim.get_ball_true_local_pos_vel()
        vx = float(vel_true[0])
        speed = float(np.linalg.norm(vel_true))
        if side == "home":
            toward_enemy = vx > 0.05
        else:
            toward_enemy = vx < -0.05
        return float(speed * 0.1) if toward_enemy else 0.0

    def _get_rewards(self, event: Optional[str], contacts: Dict[str, bool]) -> Dict[str, float]:
        home = 0.0
        away = 0.0
        dense_home = self._dense_alignment_reward("home")
        dense_away = self._dense_alignment_reward("away")

        kick_home = self._kick_velocity_reward("home", contacts.get("home", False))
        kick_away = self._kick_velocity_reward("away", contacts.get("away", False))

        comps = {
            "home": {
                "dense_block": dense_home,
                "kick_velocity": kick_home,
                "is_goal": 1.0 if event == "away_goal" else 0.0,
                "allowed_goal": 1.0 if event == "home_goal" else 0.0,
                "is_blocking": 1.0 if event == "home_block" else 0.0,
            },
            "away": {
                "dense_block": dense_away,
                "kick_velocity": kick_away,
                "is_goal": 1.0 if event == "home_goal" else 0.0,
                "allowed_goal": 1.0 if event == "away_goal" else 0.0,
                "is_blocking": 1.0 if event == "away_block" else 0.0,
            },
        }

        # home = comps["home"]["dense_block"] + comps["home"]["kick_velocity"]
        # away = comps["away"]["dense_block"] + comps["away"]["kick_velocity"]

        if event == "home_goal":
            away += 10.0
            home -= 10.0
        elif event == "away_goal":
            home += 10.0
            away -= 10.0
        elif event == "home_block":
            home += 0.5
            away -= 0.5
        elif event == "away_block":
            away += 0.5
            home -= 0.5
        elif event == "out":
            home -= 1.0
            away -= 1.0

        self._last_reward_components = comps
        return {"home": float(home), "away": float(away)}

    def _get_info(self, event: Optional[str]) -> Dict[str, Any]:
        pos_true, vel_true = self.sim.get_ball_true_local_pos_vel()
        player_home = self.sim.get_player_center_local()
        player_away = self.sim.get_opponent_player_center_local()
        slider_home, kicker_home = self.sim.get_joint_positions()
        slider_away, kicker_away = self.sim.get_opponent_joint_positions()

        return {
            "event": event,
            "dense_home": float(self._dense_alignment_reward("home")),
            "dense_away": float(self._dense_alignment_reward("away")),
            "ball_true_pos": pos_true.astype(np.float32),
            "ball_true_vel": vel_true.astype(np.float32),
            "ball_est_pos": self._est_pos.astype(np.float32),
            "ball_est_vel": self._est_vel.astype(np.float32),
            "player_home_center": player_home.astype(np.float32),
            "player_away_center": player_away.astype(np.float32) if player_away is not None else None,
            "slider_home": float(slider_home),
            "kicker_home": float(kicker_home),
            "slider_away": float(slider_away) if slider_away is not None else None,
            "kicker_away": float(kicker_away) if kicker_away is not None else None,
            "reward_components": getattr(self, "_last_reward_components", None),
        }

    # ----------------------------
    # Helpers
    # ----------------------------

    def _interp_action(self, a_norm: float, limits) -> float:
        if limits is None:
            raise RuntimeError("Opponent joint limits not available; update URDF.")
        a_clipped = float(np.clip(a_norm, -1.0, 1.0))
        return limits.lower + (a_clipped + 1.0) * 0.5 * (limits.upper - limits.lower)

    def _interp_velocity(self, a_norm: float, limits) -> float:
        if limits is None:
            raise RuntimeError("Opponent joint limits not available; update URDF.")
        a_clipped = float(np.clip(a_norm, -1.0, 1.0))
        # Map [-1,1] -> [0, max_vel]; negative still means slow.
        return float(abs(a_clipped)) * float(limits.velocity)


if __name__ == "__main__":
    # quick smoke test: run random actions (single env rollout)
    env = FoosballVersusEnv(render_mode="human", seed=0, policy_hz=200.0, sim_hz=1000, serve_side="away")
    obs, info = env.reset()
    while True:
        action = {
            "home": env.action_space["home"].sample(),
            "away": env.action_space["away"].sample(),
        }
        obs, rewards, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
