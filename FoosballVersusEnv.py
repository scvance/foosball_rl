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
        time_step: float = 1.0 / 240.0,
        action_repeat: int = 8,
        max_episode_steps: Optional[int] = None,
        max_episode_time: float = 30.0,
        seed: Optional[int] = None,
        num_substeps: int = 8,
        # shot params
        speed_min: float = 0.3,
        speed_max: float = 4.0,
        bounce_prob: float = 0.25,
        serve_side: str = "random",  # 'home' | 'away' | 'random'
        # camera noise
        cam_noise_std: Tuple[float, float, float] = (0.002, 0.002, 0.002),
    ):
        super().__init__()

        if render_mode not in ("human", "none"):
            raise ValueError("render_mode must be 'human' or 'none'")

        self.render_mode = render_mode
        self.dt = float(time_step)
        self.action_repeat = int(action_repeat)
        if max_episode_steps is None:
            # Derive steps from desired wall-clock play time.
            dt_eff = float(time_step) * float(action_repeat)
            self.max_episode_steps = int(math.ceil(max_episode_time / max(dt_eff, 1e-9)))
        else:
            self.max_episode_steps = int(max_episode_steps)

        self.speed_min = float(speed_min)
        self.speed_max = float(speed_max)
        self.bounce_prob = float(bounce_prob)
        self.serve_side = serve_side

        self.cam_noise_std = np.array(cam_noise_std, dtype=np.float32)

        # spaces
        # Action per side: [slider_pos_norm, slider_vel_norm, kicker_pos_norm, kicker_vel_norm] in [-1,1]
        act_box = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        # Observation per side: ball est pos/vel/pred (9) + own joints pos/vel (4) + opp joints pos/vel (4) = 17
        obs_box = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)
        self.action_space = spaces.Dict({"home": act_box, "away": act_box})
        self.observation_space = spaces.Dict({"home": obs_box, "away": obs_box})

        # sim
        self.sim = _FoosballSimCore(
            use_gui=(render_mode == "human"),
            time_step=self.dt,
            seed=seed,
            ball_restitution=0.5,
            table_restitution=0.5,
            wall_restitution=0.5,
            add_wall_catchers=False,
            num_substeps=int(num_substeps),
        )

        # Ensure opponent joints exist (URDF must define opponent_slider/opponent_kicker).
        if self.sim.opponent_slider_idx is None or self.sim.opponent_kicker_idx is None:
            raise RuntimeError(
                "FoosballVersusEnv expects URDF joints 'opponent_slider' and 'opponent_kicker'. "
                "Update the URDF before using this env."
            )

        # estimator state
        self._est_pos = np.zeros(3, dtype=np.float32)
        self._est_vel = np.zeros(3, dtype=np.float32)
        self._pred_pos = np.zeros(3, dtype=np.float32)
        self._vel_win = deque(maxlen=3)
        self._last_dt_eff = float(self.dt) * float(self.action_repeat)

        # episode bookkeeping
        self._episode_step = 0
        self._terminated_event: Optional[str] = None  # "home_goal" | "away_goal" | "home_block" | "away_block" | "out"

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
        for _ in range(self.action_repeat):
            self.sim.step_sim(1)

            # Update estimator every sim step using per-step dt for smoother debug markers
            self._update_estimator(first=False, dt_scale=1.0)
            self.sim.set_estimated_ball_state(self._est_pos, self._est_vel)

            if self.render_mode == "human":
                time.sleep(self.dt)

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

        obs = np.concatenate(
            [
                self._est_pos.astype(np.float32),  # 3
                self._est_vel.astype(np.float32),  # 3
                self._pred_pos.astype(np.float32),  # 3
                own_joints,  # 4
                opp_joints,  # 4
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

    def _predict_goal_intercept_est(self, side: str) -> Optional[Tuple[float, float]]:
        """
        Predicts (y,z) at the player plane (slider x) using estimated pos/vel.
        Handles a single Y-wall bounce; returns None if not heading toward the goal.
        """
        if self.sim.ball_id is None:
            return None

        x, y, z = map(float, self._est_pos)
        vx, vy, vz = map(float, self._est_vel)

        if side == "home":
            player_x = float(self.sim.get_player_center_local()[0])
        else:
            opp_center = self.sim.get_opponent_player_center_local()
            if opp_center is None:
                player_x = float(self.sim.goal_rect_x_away)
            else:
                player_x = float(opp_center[0])

        if side == "home" and vx >= -1e-4:
            return None
        if side == "away" and vx <= 1e-4:
            return None

        y_min = float(self.sim.table_min_local[1] + self.sim.ball_radius)
        y_max = float(self.sim.table_max_local[1] - self.sim.ball_radius)

        for _ in range(2):  # allow up to one bounce
            t_plane = (player_x - x) / vx
            if t_plane <= 0.0:
                return None

            t_wall = float("inf")
            if abs(vy) > 1e-6:
                if vy > 0:
                    t_wall = (y_max - y) / vy
                else:
                    t_wall = (y_min - y) / vy

            if 0.0 < t_wall < t_plane:
                # Bounce before reaching goal plane
                x += vx * t_wall
                y = y_max if vy > 0 else y_min
                z += vz * t_wall
                vy = -vy
                continue

            y_hit = y + vy * t_plane
            z_hit = z + vz * t_plane
            return (y_hit, z_hit)

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

        y_pred, z_pred = intercept

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

        home = comps["home"]["dense_block"] + comps["home"]["kick_velocity"]
        away = comps["away"]["dense_block"] + comps["away"]["kick_velocity"]

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
    env = FoosballVersusEnv(render_mode="human", seed=0, action_repeat=2)
    obs, info = env.reset()
    while True:
        action = {
            "home": env.action_space["home"].sample(),
            "away": env.action_space["away"].sample(),
        }
        obs, rewards, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
