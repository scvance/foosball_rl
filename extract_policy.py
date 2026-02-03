"""
Extract the underlying PyTorch policy from a trained SB3 SAC model.

Usage:
    python extract_policy.py --model_path ./path/to/model.zip --output_path ./policy.pt
"""

import argparse
import torch
import numpy as np
from stable_baselines3 import SAC


def extract_actor_network(model_path: str, output_path: str = None):
    """
    Extract the actor (policy) network from a trained SAC model.

    Returns the actor network and optionally saves it.
    """
    # Load the SB3 model
    model = SAC.load(model_path, device="cpu")

    # The actor network is in model.policy.actor
    # For SAC, this is an MLP that outputs mean and log_std for the action distribution
    actor = model.policy.actor

    # Get the deterministic action network (mu network)
    # SAC's actor has: features_extractor -> latent_pi -> mu (and log_std)

    print("=== SAC Policy Structure ===")
    print(f"Policy class: {type(model.policy)}")
    print(f"Actor class: {type(actor)}")
    print(f"\nActor network:")
    print(actor)

    # Get observation and action dimensions
    obs_dim = model.observation_space.shape[0]
    act_dim = model.action_space.shape[0]

    print(f"\n=== Dimensions ===")
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {act_dim}")

    # Test forward pass
    print(f"\n=== Testing Forward Pass ===")
    test_obs = torch.randn(1, obs_dim)

    # Get action from the actor (deterministic mean)
    with torch.no_grad():
        # For MlpPolicy, features_extractor is identity, so pass obs directly
        mean_action = actor.mu(actor.latent_pi(test_obs))
        action = torch.tanh(mean_action)  # Squash to [-1, 1]

    print(f"Input shape: {test_obs.shape}")
    print(f"Output shape: {action.shape}")
    print(f"Output (actions): {action}")

    # Extract just the actor state dict
    actor_state_dict = actor.state_dict()

    if output_path:
        torch.save({
            'actor_state_dict': actor_state_dict,
            'obs_dim': obs_dim,
            'act_dim': act_dim,
            'policy_class': str(type(model.policy)),
        }, output_path)
        print(f"\nSaved actor state dict to: {output_path}")

    return actor, obs_dim, act_dim


def create_standalone_actor(model_path: str, output_path: str = None):
    """
    Create a standalone PyTorch module that can be used for inference
    without SB3 dependencies.
    """
    import torch.nn as nn

    # Load model
    model = SAC.load(model_path, device="cpu")

    obs_dim = model.observation_space.shape[0]
    act_dim = model.action_space.shape[0]

    # Get the network architecture from SB3's actor
    # Default SB3 MlpPolicy uses [256, 256] hidden layers
    actor = model.policy.actor

    # Build a simple sequential model that matches the actor's mu network
    # SB3's actor structure: features_extractor -> latent_pi -> mu

    # For MlpPolicy with default settings:
    # - features_extractor is just a Flatten (identity for Box spaces)
    # - latent_pi is the MLP: Linear -> ReLU -> Linear -> ReLU
    # - mu is Linear(hidden, act_dim)

    # Extract the actual layer sizes by inspecting the model
    latent_pi = actor.latent_pi
    mu = actor.mu

    print("\n=== Building Standalone Actor ===")
    print(f"Latent pi: {latent_pi}")
    print(f"Mu layer: {mu}")

    # Create standalone network
    class StandaloneActor(nn.Module):
        def __init__(self, obs_dim, act_dim, hidden_dims=[256, 256]):
            super().__init__()

            layers = []
            prev_dim = obs_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, act_dim))
            layers.append(nn.Tanh())  # SAC uses tanh to bound actions to [-1, 1]

            self.net = nn.Sequential(*layers)

        def forward(self, obs):
            return self.net(obs)

    # Create and load weights
    standalone = StandaloneActor(obs_dim, act_dim)

    # Copy weights from SB3 actor
    with torch.no_grad():
        # Copy latent_pi weights (the hidden layers)
        standalone.net[0].weight.copy_(latent_pi[0].weight)
        standalone.net[0].bias.copy_(latent_pi[0].bias)
        standalone.net[2].weight.copy_(latent_pi[2].weight)
        standalone.net[2].bias.copy_(latent_pi[2].bias)
        # Copy mu weights (output layer)
        standalone.net[4].weight.copy_(mu.weight)
        standalone.net[4].bias.copy_(mu.bias)

    # Verify the standalone model produces same outputs
    print("\n=== Verifying Standalone Model ===")
    test_obs = torch.randn(1, obs_dim)

    with torch.no_grad():
        # SB3 actor output - for MlpPolicy, features_extractor is identity
        sb3_mean = actor.mu(actor.latent_pi(test_obs))
        sb3_action = torch.tanh(sb3_mean)

        # Standalone output
        standalone_action = standalone(test_obs)

    print(f"SB3 action: {sb3_action}")
    print(f"Standalone action: {standalone_action}")
    print(f"Max difference: {(sb3_action - standalone_action).abs().max().item():.2e}")

    if output_path:
        torch.save({
            'model_state_dict': standalone.state_dict(),
            'obs_dim': obs_dim,
            'act_dim': act_dim,
        }, output_path)
        print(f"\nSaved standalone actor to: {output_path}")

    return standalone, obs_dim, act_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to SB3 model .zip")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save extracted policy")
    parser.add_argument("--standalone", action="store_true", help="Create standalone PyTorch model")
    args = parser.parse_args()

    if args.standalone:
        actor, obs_dim, act_dim = create_standalone_actor(args.model_path, args.output_path)
    else:
        actor, obs_dim, act_dim = extract_actor_network(args.model_path, args.output_path)

    print("\n=== Final Verification ===")
    print(f"✓ Input dimension: {obs_dim}")
    print(f"✓ Output dimension: {act_dim}")


if __name__ == "__main__":
    main()
