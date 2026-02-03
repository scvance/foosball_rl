import argparse
import numpy as np
import ray
from ray.rllib.algorithms.sac import SACConfig
from FoosballVersusEnv import FoosballVersusEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    ray.init()

    # 1) Load the training config embedded in the checkpoint
    config = SACConfig.from_checkpoint(args.checkpoint)

    # 2) Force "single-process inference" (no workers/actors)
    # Old API stack uses rollout workers; new stack uses env runners.
    # We'll set both defensively.
    config = config.resources(num_gpus=0)
    config = config.env_runners(num_env_runners=0)           # new stack (harmless if unused)

    # 3) Build algorithm locally and restore weights/state
    algo = config.build()
    algo.restore(args.checkpoint)

    # 4) Your GUI env lives ONLY in the main process
    env = FoosballVersusEnv(render_mode="human")

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        while not done:
            a_home = algo.compute_single_action(obs["home"], policy_id="shared_policy")
            a_away = algo.compute_single_action(obs["away"], policy_id="shared_policy")
            obs, rewards, terminated, truncated, info = env.step({"home": a_home, "away": a_away})
            done = terminated or truncated

    env.close()
    ray.shutdown()

if __name__ == "__main__":
    main()
