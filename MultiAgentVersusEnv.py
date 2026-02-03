from ray.rllib.env.multi_agent_env import MultiAgentEnv
from FoosballVersusEnv import FoosballVersusEnv
from dataclasses import dataclass
import numpy as np
from gymnasium import spaces

class FoosballSelfPlayEnv(MultiAgentEnv):
    def __init__(self, config):
        if isinstance(config, dict):
            config = FoosballSelfPlayEnvConfig(**config)
        super().__init__()
        self.possible_agents = ["home", "away"]
        self.env = FoosballVersusEnv(
            render_mode=config.render_mode,
            policy_hz=config.policy_hz,
            sim_hz=config.sim_hz,
            max_episode_steps=config.max_episode_steps,
            seed=config.seed,
            num_substeps=config.num_substeps,
            speed_min=config.speed_min,
            speed_max=config.speed_max,
            bounce_prob=config.bounce_prob,
            cam_noise_std=(config.cam_noise_std, config.cam_noise_std, config.cam_noise_std),
            slider_vel_cap_mps=config.slider_vel_cap_mps,
            kicker_vel_cap_rads=config.kicker_vel_cap_rads,
            real_time_gui=config.real_time_gui,
            serve_side=config.serve_side,
            )
        
        self.action_spaces = {
            "home": self.env.action_space['home'],
            "away": self.env.action_space['away'],
        }
        self.observation_spaces = {
            "home": self.env.observation_space['home'],
            "away": self.env.observation_space['away'],
        }

        self.action_space = self.env.action_space['home']
        self.observation_space = self.env.observation_space['home']

        
    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return {
            "home": obs['home'],
            "away": obs['away'],
        }, info
    
    def step(self, action_dict):
        obs_dict, rewards_dict, terminated, truncated, info_dict = self.env.step(action_dict)
        terminated_dict = {
            "home": terminated,
            "away": terminated,
            "__all__": terminated,
        }
        truncated_dict = {
            "home": truncated,
            "away": truncated,
            "__all__": truncated,
        }
        return obs_dict, rewards_dict, terminated_dict, truncated_dict, {}

@dataclass
class FoosballSelfPlayEnvConfig:
    def __init__(self, render_mode: str, policy_hz: float, sim_hz: int, max_episode_steps: int, seed: int, num_substeps: int, speed_min: float, speed_max: float, bounce_prob: float, cam_noise_std: float, slider_vel_cap_mps: float, kicker_vel_cap_rads: float, real_time_gui: bool, serve_side: str):
        self.render_mode = render_mode
        self.policy_hz = policy_hz
        self.sim_hz = sim_hz
        self.max_episode_steps = max_episode_steps
        self.seed = seed
        self.num_substeps = num_substeps
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.bounce_prob = bounce_prob
        self.cam_noise_std = cam_noise_std
        self.slider_vel_cap_mps = slider_vel_cap_mps
        self.kicker_vel_cap_rads = kicker_vel_cap_rads
        self.real_time_gui = real_time_gui
        self.serve_side = serve_side

    render_mode: str = "none"
    policy_hz: float = 200.0
    sim_hz: int = 1000
    max_episode_steps: int = 1000
    seed: int = 0
    num_substeps: int = 8
    speed_min: float = 2.0
    speed_max: float = 15.0
    bounce_prob: float = 0.25
    cam_noise_std: float = 0.0
    slider_vel_cap_mps: float = 15.0
    kicker_vel_cap_rads: float = 170.0
    real_time_gui: bool = False
    serve_side: str = "random"