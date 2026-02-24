"""
ShootoutVersusEnv.py

Two-player self-play environment for the Shootout robot.
Mirrors the FoosballVersusEnv pattern exactly.

Action per side: [handle_pos, handle_vel, paddle_vel]  in [-1, 1]
  - handle_pos : maps to handle joint limits (lateral position)
  - handle_vel : maps to [0, handle_vel_cap] (speed cap for position control)
  - paddle_vel : maps to [-paddle_vel_cap, +paddle_vel_cap] (angular velocity)

Observation per side: 21 dims
  ball est pos/vel/pred (9) + own joints (4) + opp joints (4) + intercept (4)

MIRRORING FOR SINGLE-POLICY TRAINING:
  - 'away' obs are flipped so both sides see the table from the same perspective.
  - 'away' actions are un-mirrored: handle_pos and paddle_vel are negated.
  - A single policy trained with self-play can play either side.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from foosball_envs.ShootoutSimCore import _ShootoutSimCore


class ShootoutVersusEnv(gym.Env):

    metadata = {"render_modes": ["human", "none"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str = "none",
        # control rates
        policy_hz: float = 30.0,
        sim_hz: int = 240,
        max_episode_steps: int = 200,
        seed: Optional[int] = None,
        # spawn
        serve_mode: str = "random_fire",  # 'random_fire' | 'corner' | 'random'
        # camera noise
        cam_noise_std: Tuple[float, float, float] = (0.002, 0.002, 0.002),
        # actuator caps
        handle_vel_cap_mps: float = 10.0,
        paddle_vel_cap_rads: float = 20.0,
        # physics
        ball_restitution: float = 0.30,
        wall_restitution: float = 0.85,
        paddle_restitution: float = 0.85,
        num_substeps: int = 1,
        # pacing
        real_time_gui: bool = True,
    ):
        super().__init__()

        if render_mode not in ("human", "none"):
            raise ValueError("render_mode must be 'human' or 'none'")

        self.render_mode = render_mode
        self.serve_mode = serve_mode

        self.policy_hz = float(policy_hz)
        self.sim_hz = int(sim_hz)
        self.dt_sim = 1.0 / self.sim_hz
        self.dt_policy = 1.0 / self.policy_hz
        self.steps_per_policy = max(1, int(round(self.dt_policy / self.dt_sim)))
        self.dt_eff = self.steps_per_policy * self.dt_sim

        self.max_episode_steps = int(max_episode_steps)
        self.cam_noise_std = np.array(cam_noise_std, dtype=np.float32)
        self.handle_vel_cap_mps = float(handle_vel_cap_mps)
        self.paddle_vel_cap_rads = float(paddle_vel_cap_rads)
        self.real_time_gui = bool(real_time_gui)

        # Action: [handle_pos, handle_vel, paddle_vel] in [-1, 1]
        act_box = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Dict({"home": act_box, "away": act_box})

        # Observation: 21 dims
        obs_box = spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)
        self.observation_space = spaces.Dict({"home": obs_box, "away": obs_box})

        self.sim = _ShootoutSimCore(
            use_gui=(render_mode == "human"),
            time_step=self.dt_sim,
            seed=seed,
            num_substeps=num_substeps,
            ball_restitution=ball_restitution,
            wall_restitution=wall_restitution,
            paddle_restitution=paddle_restitution,
        )

        # Ball estimator state
        self._est_pos = np.zeros(3, dtype=np.float32)
        self._est_vel = np.zeros(3, dtype=np.float32)
        self._pred_pos = np.zeros(3, dtype=np.float32)
        self._vel_win: deque = deque(maxlen=3)
        self._pos_hist: list = []

        # Episode bookkeeping
        self._episode_step = 0
        self._terminated_event: Optional[str] = None

        if self.sim.opponent_handle_idx is None or self.sim.opponent_paddle_idx is None:
            raise RuntimeError(
                "ShootoutVersusEnv requires opponent_handle and opponent_paddle joints."
            )

    # ── Mirroring helpers ────────────────────────────────────────────────────

    def _mirror_position(self, pos: np.ndarray) -> np.ndarray:
        """Flip x and y axes (table is centred at origin, symmetric about both)."""
        m = pos.copy().astype(np.float32)
        m[0] = -pos[0]
        m[1] = -pos[1]
        return m

    def _mirror_velocity(self, vel: np.ndarray) -> np.ndarray:
        m = vel.copy().astype(np.float32)
        m[0] = -vel[0]
        m[1] = -vel[1]
        return m

    # ── Gymnasium API ────────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._episode_step = 0
        self._terminated_event = None

        self.sim.remove_ball()
        self.sim.reset_robot_randomized()

        mode = self.serve_mode
        if mode == "random":
            mode = "random_fire" if self.np_random.random() < 0.5 else "corner"

        if mode == "random_fire":
            target = "home" if self.np_random.random() < 0.5 else "away"
            shot_info = self.sim.spawn_ball_random_fire(target=target)
        else:
            side = "home" if self.np_random.random() < 0.5 else "away"
            shot_info = self.sim.spawn_ball_corner_serve(side=side)

        self._update_estimator(first=True)
        obs = self._get_obs()
        return obs, {"shot": shot_info}

    def step(self, action: Dict[str, np.ndarray]):
        a_home = np.asarray(action["home"], dtype=np.float32).reshape(-1)
        a_away = np.asarray(action["away"], dtype=np.float32).reshape(-1)

        self._episode_step += 1

        # ── Home action ─────────────────────────────────────────────────────
        h_hp = float(np.clip(a_home[0], -1.0, 1.0))
        h_hv = float(np.clip(a_home[1], -1.0, 1.0))
        h_pv = float(np.clip(a_home[2], -1.0, 1.0))

        home_handle_target = self._interp_to_limits(h_hp, self.sim.handle_limits)
        home_handle_vel_cap = (h_hv + 1.0) * 0.5 * self.handle_vel_cap_mps
        home_paddle_vel = h_pv * self.paddle_vel_cap_rads

        # ── Away action (un-mirror) ──────────────────────────────────────────
        # Policy output is "as if playing home"; negate handle_pos and paddle_vel
        a_hp = float(np.clip(a_away[0], -1.0, 1.0))
        a_hv = float(np.clip(a_away[1], -1.0, 1.0))
        a_pv = float(np.clip(a_away[2], -1.0, 1.0))

        away_handle_target = self._interp_to_limits(-a_hp, self.sim.opponent_handle_limits)
        away_handle_vel_cap = (a_hv + 1.0) * 0.5 * self.handle_vel_cap_mps
        away_paddle_vel = -a_pv * self.paddle_vel_cap_rads

        self.sim.apply_action_targets_dual(
            home_handle_target, home_paddle_vel,
            away_handle_target, away_paddle_vel,
            home_handle_vel_cap=home_handle_vel_cap,
            away_handle_vel_cap=away_handle_vel_cap,
        )

        terminated = False
        truncated = False
        event = None
        out_reason = ""

        for _ in range(self.steps_per_policy):
            self.sim.step_sim(1)
            self._update_estimator(first=False, dt_scale=1.0)

            if self.render_mode == "human" and self.real_time_gui:
                time.sleep(self.dt_sim * 0.5)

            goals = self.sim.check_goal_crossings_dual()
            out, reason = self.sim.check_ball_out_of_bounds()

            if goals.get("home"):
                terminated, event = True, "home_goal"
                break
            if goals.get("away"):
                terminated, event = True, "away_goal"
                break
            if out:
                truncated, event, out_reason = True, "out", reason
                break
            if self._detect_stalled_ball():
                truncated, event = True, "stalled"
                break

        self._terminated_event = event
        rewards = self._get_rewards(event, action)

        if self._episode_step >= self.max_episode_steps:
            truncated = True

        obs = self._get_obs()
        info = self._get_info(event)
        if out_reason:
            info["out_reason"] = out_reason

        return obs, rewards, bool(terminated), bool(truncated), info

    # ── Observation ──────────────────────────────────────────────────────────

    def _detect_stalled_ball(self) -> bool:
        pos, vel = self.sim.get_ball_true_local_pos_vel()
        speed = float(np.linalg.norm(vel))
        near_goal = abs(float(pos[0])) > self.sim.goal_x - 0.15
        return (not near_goal) and speed < 0.05

    def _sample_noisy_pos(self) -> np.ndarray:
        pos, _ = self.sim.get_ball_true_local_pos_vel()
        noise = np.array([
            self.np_random.normal(0.0, float(self.cam_noise_std[i]))
            for i in range(3)
        ], dtype=np.float32)
        return pos.astype(np.float32) + noise

    def _update_estimator(self, first: bool, dt_scale: Optional[float] = None):
        pos = self._sample_noisy_pos()
        dt_eff = (self.dt_sim * dt_scale) if dt_scale is not None else self.dt_eff

        if first:
            self._est_pos = pos.copy()
            self._est_vel[:] = 0.0
            self._pred_pos = pos.copy()
            self._vel_win.clear()
            self._vel_win.append((0.0, pos.copy()))
            self._pos_hist = [(0.0, pos.copy())]
            return

        t_prev = self._vel_win[-1][0] if self._vel_win else 0.0
        t_new = t_prev + dt_eff

        noise_tol = float(max(self.cam_noise_std)) * 3.0
        y_min_wall = -(self.sim.table_y_half - self.sim.ball_radius - 0.001)
        y_max_wall = self.sim.table_y_half - self.sim.ball_radius - 0.001

        if self._pos_hist:
            last_pos = self._pos_hist[-1][1]
            base_pos = self._pos_hist[0][1]
            step_vec = pos - last_pos
            main_vec = last_pos - base_pos
            main_len = float(np.linalg.norm(main_vec))

            # Y-wall bounce detection
            y0, y1 = float(last_pos[1]), float(pos[1])
            bounce_wall_y = None
            if (y0 < y_min_wall and y1 >= y_min_wall) or (y0 > y_min_wall and y1 <= y_min_wall):
                bounce_wall_y = y_min_wall
            if (y0 < y_max_wall and y1 >= y_max_wall) or (y0 > y_max_wall and y1 <= y_max_wall):
                if bounce_wall_y is None:
                    bounce_wall_y = y_max_wall

            if bounce_wall_y is not None and abs(float(step_vec[1])) > 1e-6:
                t_hit = max(0.0, min(1.0, (bounce_wall_y - y0) / float(step_vec[1])))
                bounce_pos = last_pos + step_vec * t_hit
                dt_post = max(dt_eff * (1.0 - t_hit), 1e-6)
                self._vel_win.clear()
                self._pos_hist = [(0.0, bounce_pos.astype(np.float32))]
                self._vel_win.append((0.0, bounce_pos.astype(np.float32)))
                self._pos_hist.append((dt_post, pos.copy()))
                self._est_pos = pos.copy()
                self._est_vel = ((pos - bounce_pos) / dt_post).astype(np.float32)
                self._pred_pos = (self._est_pos + self._est_vel * self.dt_eff).astype(np.float32)
                return

            # Direction-flip / line-deviation reset
            if len(self._pos_hist) >= 2:
                prev_pos = self._pos_hist[-2][1]
                dx_p, dy_p = float(last_pos[0] - prev_pos[0]), float(last_pos[1] - prev_pos[1])
                dx_n, dy_n = float(step_vec[0]), float(step_vec[1])
                y_inner = self.sim.table_y_half - self.sim.ball_radius - 0.002
                near_y = (abs(float(last_pos[1])) > y_inner - 0.01)
                if near_y and (dy_p * dy_n < -1e-6) and abs(dy_p) > 1e-4 and abs(dy_n) > 1e-4:
                    self._vel_win.clear()
                    self._pos_hist = [(0.0, pos.copy())]
                    self._est_pos = pos.copy()
                    self._est_vel[:] = 0.0
                    self._pred_pos = pos.copy()
                    return

            travel = float(np.linalg.norm(last_pos - base_pos))
            step_len = float(np.linalg.norm(step_vec))
            dir_flip = (main_len > 1e-6 and step_len > 1e-6
                        and float(np.dot(step_vec, main_vec)) < -1e-4)
            dev = 0.0
            line_dev = False
            if travel >= 0.025 and main_len > 1e-6:
                rel = pos - base_pos
                proj = float(np.dot(rel, main_vec)) / (main_len * main_len)
                dev = float(np.linalg.norm(pos - (base_pos + proj * main_vec)))
                line_dev = dev > max(noise_tol * 2.0, noise_tol + 0.003)

            hist_dt = t_new - self._pos_hist[0][0]
            avg_speed = travel / max(hist_dt, 1e-6)
            if avg_speed >= 0.8 and travel >= 0.045 and len(self._pos_hist) >= 3 and (dir_flip or line_dev):
                self._vel_win.clear()
                self._pos_hist = [(0.0, pos.copy())]
                self._est_pos = pos.copy()
                self._est_vel[:] = 0.0
                self._pred_pos = pos.copy()
                return

        self._vel_win.append((t_new, pos.copy()))
        self._pos_hist.append((t_new, pos.copy()))

        if len(self._pos_hist) >= 2:
            times = np.array([t for t, _ in self._pos_hist], dtype=np.float32)
            coords = np.stack([p for _, p in self._pos_hist])
            t_norm = times - times[0]
            t_mean = float(np.mean(t_norm))
            p_mean = np.mean(coords, axis=0)
            denom = float(np.sum((t_norm - t_mean) ** 2))
            if denom > 1e-8:
                slopes = np.sum((t_norm - t_mean).reshape(-1, 1) * (coords - p_mean), axis=0) / denom
                intercepts = p_mean - slopes * t_mean
                self._est_vel = slopes.astype(np.float32)
                self._est_pos = (intercepts + slopes * float(t_norm[-1])).astype(np.float32)
            else:
                self._est_pos = pos.copy()
        else:
            self._est_pos = pos.copy()

        self._pred_pos = (self._est_pos + self._est_vel * self.dt_eff).astype(np.float32)

    @staticmethod
    def _wrap_pi(angle: float) -> float:
        return ((angle + math.pi) % (2.0 * math.pi)) - math.pi

    def _predict_goal_intercept_est(self, side: str) -> Optional[Tuple[float, float, float, float]]:
        if self.sim.ball_id is None:
            return None
        x, y, z = map(float, self._est_pos)
        vx, vy, vz = map(float, self._est_vel)

        if side == "home" and vx >= -1e-4:
            return None
        if side == "away" and vx <= 1e-4:
            return None

        x_goal = -self.sim.goal_x if side == "home" else self.sim.goal_x
        y_min = -(self.sim.table_y_half - self.sim.ball_radius)
        y_max = self.sim.table_y_half - self.sim.ball_radius

        t_accum = 0.0
        for _ in range(2):
            if abs(vx) < 1e-6:
                return None
            t_goal = (x_goal - x) / vx
            if t_goal <= 0.0:
                return None
            t_wall = (y_max - y) / vy if vy > 1e-6 else ((y_min - y) / vy if vy < -1e-6 else float("inf"))
            if 0.0 < t_wall < t_goal:
                x += vx * t_wall
                y = y_max if vy > 0 else y_min
                z += vz * t_wall
                vy = -vy
                t_accum += t_wall
                continue
            t_accum += t_goal
            return (y + vy * t_goal, z + vz * t_goal, x_goal, t_accum)
        return None

    def _get_obs_for_side(self, side: str) -> np.ndarray:
        (h_home, k_home), (vh_home, vk_home) = self.sim.get_joint_positions_and_vels()
        (h_away, k_away), (vh_away, vk_away) = self.sim.get_opponent_joint_positions_and_vels()

        if side == "home":
            own = np.array([self._wrap_pi(k_home), h_home, vk_home, vh_home], dtype=np.float32)
            opp = np.array([
                self._wrap_pi(k_away) if k_away is not None else 0.0,
                h_away if h_away is not None else 0.0,
                vk_away if vk_away is not None else 0.0,
                vh_away if vh_away is not None else 0.0,
            ], dtype=np.float32)
            est_pos = self._est_pos.copy()
            est_vel = self._est_vel.copy()
            pred_pos = self._pred_pos.copy()
            intercept = self._predict_goal_intercept_est("home")
            y_i, z_i, x_i, t_i = intercept if intercept else (0.0, 0.0, 0.0, 0.0)
        else:
            # Mirror: flip sign of handle_pos and paddle_vel so away looks like home
            own = np.array([
                self._wrap_pi(-(k_away or 0.0)),
                -(h_away or 0.0),
                -(vk_away or 0.0),
                -(vh_away or 0.0),
            ], dtype=np.float32)
            opp = np.array([
                self._wrap_pi(-k_home),
                -h_home,
                -vk_home,
                -vh_home,
            ], dtype=np.float32)
            est_pos = self._mirror_position(self._est_pos)
            est_vel = self._mirror_velocity(self._est_vel)
            pred_pos = self._mirror_position(self._pred_pos)
            intercept = self._predict_goal_intercept_est("away")
            if intercept:
                y_i, z_i, x_i, t_i = intercept
                x_i = -x_i   # mirror x of interception plane
                y_i = -y_i   # mirror y of interception point
            else:
                y_i, z_i, x_i, t_i = 0.0, 0.0, 0.0, 0.0

        return np.concatenate([
            est_pos, est_vel, pred_pos,   # 9
            own, opp,                     # 8
            np.array([y_i, z_i, x_i, t_i], dtype=np.float32),  # 4
        ]).astype(np.float32)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {
            "home": self._get_obs_for_side("home"),
            "away": self._get_obs_for_side("away"),
        }

    # ── Reward ───────────────────────────────────────────────────────────────

    def _get_rewards(
        self,
        event: Optional[str],
        actions: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        home = 0.0
        away = 0.0

        # Sparse terminal
        if event == "home_goal":
            home -= 10.0
            away += 10.0
        elif event == "away_goal":
            away -= 10.0
            home += 10.0
        elif event == "out":
            home -= 0.5
            away -= 0.5

        # Contact
        if self.sim.had_contact_with_paddle(self.sim.paddle_idx):
            home += 0.1
        if self.sim.had_contact_with_paddle(self.sim.opponent_paddle_idx):
            away += 0.1

        # Ball velocity shaping: reward pushing ball toward opponent's goal
        vx = float(self._est_vel[0])
        home += 0.01 * vx    # positive vx = toward away goal (+x)
        away += 0.01 * (-vx) # negative vx = toward home goal (−x)

        # Tracking: reward when ball heads toward your goal and you're aligned
        ball_y = float(self._est_pos[1])
        tracking_scale = 0.05
        tracking_max = 0.10

        if vx < -1.0:  # ball heading toward home
            p_home = self.sim.get_player_center_local()
            err = abs(float(p_home[1]) - ball_y)
            home += tracking_scale * max(0.0, 1.0 - err / tracking_max)

        if vx > 1.0:  # ball heading toward away
            p_away = self.sim.get_opponent_player_center_local()
            if p_away is not None:
                err = abs(float(p_away[1]) - ball_y)
                away += tracking_scale * max(0.0, 1.0 - err / tracking_max)

        # Action penalty
        a_h = np.asarray(actions["home"], dtype=np.float32)
        a_a = np.asarray(actions["away"], dtype=np.float32)
        home -= 0.05 * float(np.linalg.norm(a_h[1:]))  # penalise vel/spin commands
        away -= 0.05 * float(np.linalg.norm(a_a[1:]))

        return {"home": float(home), "away": float(away)}

    # ── Info ─────────────────────────────────────────────────────────────────

    def _get_info(self, event: Optional[str]) -> Dict[str, Any]:
        pos, vel = self.sim.get_ball_true_local_pos_vel()
        h_home, k_home = self.sim.get_joint_positions()
        h_away, k_away = self.sim.get_opponent_joint_positions()
        return {
            "event": event,
            "ball_true_pos": pos.astype(np.float32),
            "ball_true_vel": vel.astype(np.float32),
            "ball_est_pos": self._est_pos.copy(),
            "ball_est_vel": self._est_vel.copy(),
            "handle_home": float(h_home),
            "paddle_home": float(k_home),
            "handle_away": float(h_away) if h_away is not None else None,
            "paddle_away": float(k_away) if k_away is not None else None,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _interp_to_limits(self, a_norm: float, limits) -> float:
        a = float(np.clip(a_norm, -1.0, 1.0))
        return limits.lower + (a + 1.0) * 0.5 * (limits.upper - limits.lower)

    def close(self):
        self.sim.close()

    def render(self):
        pass  # PyBullet GUI handles rendering natively


if __name__ == "__main__":
    env = ShootoutVersusEnv(render_mode="human", seed=0, serve_mode="random_fire")
    obs, info = env.reset()
    print("obs home:", obs["home"].shape)
    print("obs away:", obs["away"].shape)
    while True:
        action = {k: env.action_space[k].sample() for k in ("home", "away")}
        obs, rewards, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"Episode: {info.get('event')}  rewards={rewards}")
            obs, info = env.reset()
