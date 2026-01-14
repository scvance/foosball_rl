from typing import Optional, Tuple, Dict, Any
from collections import deque
import time

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from FoosballSimCore import _FoosballSimCore


class FoosballGoalieEnv(gym.Env):
    """
    Gymnasium environment for Stable-Baselines3.
    - Action: [a_slider, a_kicker] in [-1, 1]
    - Observation:
        [ball_est_pos_local(3),
         ball_est_vel_local(3)  (estimated from last 3 frames),
         ball_pred_pos_local(3) (pos + vel * dt_pred),
         kicker_angle,
         slider_pos]
      => total = 3 + 3 + 3 + 2 = 11
    - Reward: dense alignment (0..1) + sparse (+0.5 block, -10 goal/out)
    """

    metadata = {"render_modes": ["human", "none"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str = "none",
        time_step: float = 1.0 / 240.0,
        action_repeat: int = 8,
        max_episode_steps: int = 1500,
        seed: Optional[int] = None,
        num_substeps: int = 8,
        # shot params
        speed_min: float = 2.0,
        speed_max: float = 10.0,
        bounce_prob: float = 0.25,
        # camera noise
        cam_noise_std: Tuple[float, float, float] = (0.002, 0.002, 0.002),
    ):
        super().__init__()

        if render_mode not in ("human", "none"):
            raise ValueError("render_mode must be 'human' or 'none'")

        self.render_mode = render_mode
        self.dt = float(time_step)
        self.action_repeat = int(action_repeat)
        self.max_episode_steps = int(max_episode_steps)

        self.speed_min = float(speed_min)
        self.speed_max = float(speed_max)
        self.bounce_prob = float(bounce_prob)

        self.cam_noise_std = np.array(cam_noise_std, dtype=np.float32)

        # spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # ✅ updated obs: 11 dims (pos 3 + vel 3 + pred pos 3 + joints 2)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)

        # sim
        self.sim = _FoosballSimCore(
            use_gui=(render_mode == "human"),
            time_step=self.dt,
            seed=seed,
            ball_restitution=0.5,
            table_restitution=0.5,
            wall_restitution=0.5,
            add_wall_catchers=True,
            num_substeps=int(num_substeps),
        )

        # estimator state
        self._est_pos = np.zeros(3, dtype=np.float32)
        self._est_vel = np.zeros(3, dtype=np.float32)

        # ✅ predicted position (new obs)
        self._pred_pos = np.zeros(3, dtype=np.float32)

        # ✅ use a 3-frame window for velocity estimation
        # stores (t, pos) for last 3 samples
        self._vel_win = deque(maxlen=3)

        # ✅ track last effective dt used for prediction horizon
        self._last_dt_eff = float(self.dt) * float(self.action_repeat)

        # episode bookkeeping
        self._episode_step = 0
        self._terminated_event: Optional[str] = None  # "goal" or "block" or None
        self._goal_count = 0
        self._block_count = 0
        self._out_count = 0
        self._intercept_available = False

        # HUD (only renders when GUI is enabled)
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

        # reset sim
        self.sim.remove_ball()
        self.sim.reset_robot()

        # spawn a new shot
        shot_info = self.sim.spawn_shot_random(self.speed_min, self.speed_max, self.bounce_prob)

        # init estimator from noisy position
        self._update_estimator(first=True)
        self.sim.set_estimated_ball_state(self._est_pos, self._est_vel)

        obs = self._get_obs()
        info = {"shot": shot_info}
        self._update_scoreboard(highlight=None)
        return obs, info

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape != (2,):
            raise ValueError(f"Action must have shape (2,), got {a.shape}")

        self._episode_step += 1

        # map action [-1,1] to joint ranges
        a_slider = float(np.clip(a[0], -1.0, 1.0))
        a_kicker = float(np.clip(a[1], -1.0, 1.0))

        slider_target = self.sim.slider_limits.lower + (a_slider + 1.0) * 0.5 * (self.sim.slider_limits.upper - self.sim.slider_limits.lower)
        kicker_target = self.sim.kicker_limits.lower + (a_kicker + 1.0) * 0.5 * (self.sim.kicker_limits.upper - self.sim.kicker_limits.lower)

        self.sim.apply_action_targets(slider_target, kicker_target)

        terminated = False
        truncated = False
        event = None
        out_reason = ""

        for _ in range(self.action_repeat):
            self.sim.step_sim(1)

            # Update estimator every sim step using per-step dt for smoother debug markers
            self._update_estimator(first=False, dt_scale=1.0)
            self.sim.set_estimated_ball_state(self._est_pos, self._est_vel)

            if self.render_mode == "human":
                time.sleep(self.dt)

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

        # choose effective dt for this estimator update
        if dt_scale is None:
            dt_eff = float(self.dt) * float(self.action_repeat)
        else:
            dt_eff = float(self.dt) * float(dt_scale)
        self._last_dt_eff = dt_eff

        if first:
            self._est_pos = pos.copy()
            self._est_vel[:] = 0.0
            self._pred_pos = pos.copy()

            self._vel_win.clear()
            self._vel_win.append((0.0, pos.copy()))
            return

        # advance time in the vel window
        t_prev = self._vel_win[-1][0] if len(self._vel_win) > 0 else 0.0
        t_new = t_prev + dt_eff
        self._vel_win.append((t_new, pos.copy()))

        # velocity estimate from 3 frames (preferred)
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

        self._est_pos = pos.copy()
        self._est_vel = vel.astype(np.float32)

        # predicted next position (line model using velocity)
        self._pred_pos = (self._est_pos + self._est_vel * float(self._last_dt_eff)).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        slider_pos, kicker_pos = self.sim.get_joint_positions()
        obs = np.concatenate(
            [
                self._est_pos.astype(np.float32),     # 3
                self._est_vel.astype(np.float32),     # 3 (3-frame)
                self._pred_pos.astype(np.float32),    # 3 (new)
                np.array([kicker_pos, slider_pos], dtype=np.float32),  # 2
            ],
            axis=0,
        )
        return obs

    # ----------------------------
    # Reward
    # ----------------------------

    def _predict_goal_intercept_est(self) -> Optional[Tuple[float, float]]:
        """
        Predicts (y,z) at the goal plane using estimated pos/vel.
        Handles a single Y-wall bounce; returns None if not heading toward the goal.
        """
        if self.sim.ball_id is None:
            return None

        x, y, z = map(float, self._est_pos)
        vx, vy, vz = map(float, self._est_vel)

        if vx >= -1e-4:
            return None

        x_goal = float(self.sim.goal_rect_x)
        y_min = float(self.sim.table_min_local[1] + self.sim.ball_radius)
        y_max = float(self.sim.table_max_local[1] - self.sim.ball_radius)

        for _ in range(2):  # allow up to one bounce
            t_goal = (x_goal - x) / vx  # vx negative
            if t_goal <= 0.0:
                return None

            t_wall = float("inf")
            if abs(vy) > 1e-6:
                if vy > 0:
                    t_wall = (y_max - y) / vy
                else:
                    t_wall = (y_min - y) / vy

            if 0.0 < t_wall < t_goal:
                # Bounce before reaching goal plane
                x += vx * t_wall
                y = y_max if vy > 0 else y_min
                z += vz * t_wall
                vy = -vy
                continue

            y_hit = y + vy * t_goal
            z_hit = z + vz * t_goal
            return (y_hit, z_hit)

        return None

    def _dense_alignment_reward(self) -> float:
        """
        Dense 0..1 reward for being aligned with the predicted (noisy) intercept line at x = goal plane.
        We predict y at the time the noisy estimate reaches x_goal, then compare to player y coverage.
        """
        intercept = self._predict_goal_intercept_est()
        self._debug_goal_intercept(intercept)
        if intercept is None:
            self._intercept_available = False
            return 0.0

        self._intercept_available = True
        y_pred, z_pred = intercept

        # player position / coverage
        player_c = self.sim.get_player_center_local()
        y_p = float(player_c[1])

        y_half = float(self.sim.get_player_y_halfwidth())
        denom = y_half + float(self.sim.ball_radius)

        y_err = abs(y_pred - y_p)
        r_close = max(0.0, 1.0 - (y_err / max(denom, 1e-6)))
        r_tight = max(0.0, 1.0 - (y_err / max(0.4 * denom, 1e-6)))

        # optional: gate by z being inside approximate mouth height
        z_ok = (self.sim.goal_rect_z_min - 0.02) <= z_pred <= (self.sim.goal_rect_z_max + 0.05)

        if not z_ok:
            return 0.0

        # Blend close and "right on" tiers
        return float(0.5 * r_close + 0.5 * r_tight) * 0.1

    def _debug_goal_intercept(self, intercept: Optional[Tuple[float, float]]) -> None:
        """
        Visual marker (GUI only) for the predicted intercept used by the dense reward.
        """
        if self.render_mode != "human":
            return
        if intercept is None:
            self.sim.set_goal_intercept_debug(None, None)
            return
        y_pred, z_pred = intercept
        self.sim.set_goal_intercept_debug(float(y_pred), float(z_pred))

    def _get_reward(self, event: Optional[str]) -> float:
        dense = self._dense_alignment_reward()
        sparse = 0.0
        if event == "block":
            sparse += 0.5
        elif event == "goal":
            sparse -= 10.0
        elif event == "out":
            sparse -= 10.0  # penalty for ball going out of bounds
        return float(dense + sparse)

    def _get_info(self, event: Optional[str]) -> Dict[str, Any]:
        pos_true, vel_true = self.sim.get_ball_true_local_pos_vel()
        player_c = self.sim.get_player_center_local()
        slider_pos, kicker_pos = self.sim.get_joint_positions()

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
            "intercept_available": bool(self._intercept_available),
        }

    def _update_scoreboard(self, highlight: Optional[str]):
        # Renders a large counter above the goal when the GUI is active.
        if hasattr(self, "sim"):
            try:
                self.sim.update_scoreboard_text(self._goal_count, self._block_count, self._out_count, highlight)
            except AttributeError:
                pass


if __name__ == "__main__":
    # quick smoke test: run random actions
    env = FoosballGoalieEnv(render_mode="human", seed=0, action_repeat=2)
    obs, info = env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
