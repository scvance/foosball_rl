from typing import Optional, Tuple, Dict, Any
import time

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from FoosballSimCore import _FoosballSimCore


class FoosballGoalieEnv(gym.Env):
    """
    Gymnasium environment for Stable-Baselines3.
    - Action: [a_slider, a_kicker] in [-1, 1]
    - Observation: [ball_est_pos_local(3), ball_est_vel_local(3), kicker_angle, slider_pos]
    - Reward: dense alignment (0..1) + sparse (+10 block, -10 goal)
    """

    metadata = {"render_modes": ["human", "none"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str = "none",
        time_step: float = 1.0 / 240.0,
        action_repeat: int = 8,
        max_episode_steps: int = 1500,
        seed: Optional[int] = None,
        # shot params
        speed_min: float = 2.0,
        speed_max: float = 10.0,
        bounce_prob: float = 0.25,
        # camera noise
        cam_noise_std: Tuple[float, float, float] = (0.006, 0.006, 0.002),
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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

        # sim
        self.sim = _FoosballSimCore(
            use_gui=(render_mode == "human"),
            time_step=self.dt,
            seed=seed,
            ball_restitution=0.5,
            table_restitution=0.5,
            wall_restitution=0.5,
            add_wall_catchers=True,
            num_substeps=20,
        )

        # estimator state
        self._est_pos = np.zeros(3, dtype=np.float32)
        self._est_pos_prev = np.zeros(3, dtype=np.float32)
        self._est_vel = np.zeros(3, dtype=np.float32)

        # episode bookkeeping
        self._episode_step = 0
        self._terminated_event: Optional[str] = None  # "goal" or "block" or None

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

        obs = self._get_obs()
        info = {"shot": shot_info}
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

        for _ in range(self.action_repeat):
            self.sim.step_sim(1)

            # ✅ pace GUI so motion looks smooth
            if self.render_mode == "human":
                time.sleep(self.dt)

            out, reason = self.sim.check_ball_out_of_bounds()
            if out:
                truncated = True
                event = "out"
                out_reason = reason
                break

            if self.sim.check_goal_crossing():
                terminated = True
                event = "goal"
                out_reason = ""
                break
            if self.sim.check_block_event():
                terminated = True
                event = "block"
                out_reason = ""
                break

        # update estimator once per env step (as you requested)
        self._update_estimator(first=False)

        # compute reward
        self._terminated_event = event
        reward = self._get_reward(event)

        # truncation by time limit
        if self._episode_step >= self.max_episode_steps:
            truncated = True

        obs = self._get_obs()
        info = self._get_info(event)
        if event == "out":
            info["out_reason"] = out_reason


        # If terminal, SB3 expects you to call reset() externally next.
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        # For "human" PyBullet renders in its own window.
        return None

    def close(self):
        self.sim.close()

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

    def _update_estimator(self, first: bool):
        pos = self._sample_noisy_ball_pos_local()

        if first:
            self._est_pos_prev = pos.copy()
            self._est_pos = pos.copy()
            self._est_vel[:] = 0.0
            return

        dt_eff = float(self.dt) * float(self.action_repeat)
        self._est_vel = (pos - self._est_pos_prev) / max(dt_eff, 1e-9)
        self._est_pos_prev = pos.copy()
        self._est_pos = pos.copy()

    def _get_obs(self) -> np.ndarray:
        slider_pos, kicker_pos = self.sim.get_joint_positions()
        obs = np.concatenate(
            [
                self._est_pos.astype(np.float32),
                self._est_vel.astype(np.float32),
                np.array([kicker_pos, slider_pos], dtype=np.float32),
            ],
            axis=0,
        )
        return obs

    # ----------------------------
    # Reward
    # ----------------------------

    def _dense_alignment_reward(self) -> float:
        """
        Dense 0..1 reward for being aligned with the predicted (noisy) intercept line at x = goal plane.
        We predict y at the time the noisy estimate reaches x_goal, then compare to player y coverage.
        """
        if self.sim.ball_id is None:
            return 0.0

        x_goal = float(self.sim.goal_rect_x)
        x, y, z = float(self._est_pos[0]), float(self._est_pos[1]), float(self._est_pos[2])
        vx, vy, vz = float(self._est_vel[0]), float(self._est_vel[1]), float(self._est_vel[2])

        # only if moving toward goal plane (negative x)
        if vx >= -1e-3:
            return 0.0

        t = (x_goal - x) / vx  # vx negative -> t positive
        if t <= 0.0 or t > 2.0:
            return 0.0

        y_pred = y + vy * t
        z_pred = z + vz * t

        # player position / coverage
        player_c = self.sim.get_player_center_local()
        y_p = float(player_c[1])

        y_half = float(self.sim.get_player_y_halfwidth())
        denom = y_half + float(self.sim.ball_radius)

        # linear reward in y
        y_err = abs(y_pred - y_p)
        r_y = max(0.0, 1.0 - (y_err / max(denom, 1e-6)))

        # optional: gate by z being inside approximate mouth height
        z_ok = (self.sim.goal_rect_z_min - 0.02) <= z_pred <= (self.sim.goal_rect_z_max + 0.05)
        r_z = 1.0 if z_ok else 0.0

        return float(r_y * r_z)

    def _get_reward(self, event: Optional[str]) -> float:
        dense = 0.0 # self._dense_alignment_reward()
        sparse = 0.0
        if event == "block":
            sparse += 10.0
        elif event == "goal":
            sparse -= 10.0
        elif event == "out":
            sparse -= 1.0  # penalty for ball going out of bounds
        return float(dense + sparse)

    def _get_info(self, event: Optional[str]) -> Dict[str, Any]:
        pos_true, vel_true = self.sim.get_ball_true_local_pos_vel()
        player_c = self.sim.get_player_center_local()
        slider_pos, kicker_pos = self.sim.get_joint_positions()

        return {
            "event": event,
            "dense": 0.0, # float(self._dense_alignment_reward()),
            "ball_true_pos": pos_true.astype(np.float32),
            "ball_true_vel": vel_true.astype(np.float32),
            "ball_est_pos": self._est_pos.astype(np.float32),
            "ball_est_vel": self._est_vel.astype(np.float32),
            "player_center": player_c.astype(np.float32),
            "slider_pos": float(slider_pos),
            "kicker_pos": float(kicker_pos),
        }


if __name__ == "__main__":
    # quick smoke test: run random actions
    env = FoosballGoalieEnv(render_mode="human", seed=0)
    obs, info = env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
