"""
extract_actor_and_verify_rllib_sac.py

Old RLlib stack, no env registration.

- Loads SAC policy from a policy checkpoint dir.
- Extracts actor MLP (21->256->256->6) into a pure PyTorch module and saves weights.
- Verifies:
    (1) RLlib actor output == pure actor output (6 dims: [mean(3), log_std(3)])
    (2) RLlib deterministic action == mean (first 3 dims)
    (3) RLlib stochastic actions look like samples from N(mean, exp(log_std)^2)

Run:
  python extract_actor_and_verify_rllib_sac.py
"""

import numpy as np
import torch
import torch.nn as nn
from ray.rllib.policy.policy import Policy

ckpt_dir = "/Users/samvance/vscode projects/Foosball_RL/checkpoints/policies/shared_policy"
print("Loading from:", ckpt_dir)

torch.set_grad_enabled(False)

# -----------------------------
# Load policy (no env required)
# -----------------------------
policy = Policy.from_checkpoint(ckpt_dir)
print("Policy type:", type(policy))

model = policy.model
torch_model = getattr(model, "torch_model", None) or model
print("torch_model type:", type(torch_model))

action_model = torch_model.action_model
print("action_model type:", type(action_model))

# -----------------------------
# Extract RLlib actor linears
# -----------------------------
lin1 = action_model._hidden_layers[0]._model[0]  # Linear(21->256)
lin2 = action_model._hidden_layers[1]._model[0]  # Linear(256->256)
lin3 = action_model._logits._model[0]            # Linear(256->6)

print("Actor Linear shapes:")
print("  lin1:", tuple(lin1.weight.shape), tuple(lin1.bias.shape))
print("  lin2:", tuple(lin2.weight.shape), tuple(lin2.bias.shape))
print("  lin3:", tuple(lin3.weight.shape), tuple(lin3.bias.shape))

OBS_DIM = lin1.in_features
H1 = lin1.out_features
H2 = lin2.out_features
OUT_DIM = lin3.out_features  # should be 6
if OUT_DIM % 2 != 0:
    raise RuntimeError(f"Expected even OUT_DIM for [mean,log_std], got {OUT_DIM}")

ACT_DIM = OUT_DIM // 2

# -----------------------------
# Pure torch actor MLP
# -----------------------------
class PureActorMLP(nn.Module):
    def __init__(self, obs_dim: int, h1: int, h2: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, out_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

pure_actor = PureActorMLP(OBS_DIM, H1, H2, OUT_DIM).eval()

with torch.no_grad():
    pure_actor.net[0].weight.copy_(lin1.weight)
    pure_actor.net[0].bias.copy_(lin1.bias)

    pure_actor.net[2].weight.copy_(lin2.weight)
    pure_actor.net[2].bias.copy_(lin2.bias)

    pure_actor.net[4].weight.copy_(lin3.weight)
    pure_actor.net[4].bias.copy_(lin3.bias)

torch.save(pure_actor.state_dict(), "pure_actor_state_dict.pt")
print("Saved pure actor state_dict: pure_actor_state_dict.pt")

# -----------------------------
# Helper: RLlib actor forward output (6 dims)
# -----------------------------
def rllib_actor_out(obs_batch_np: np.ndarray) -> np.ndarray:
    obs_t = torch.from_numpy(obs_batch_np).float()
    out, _state = action_model({"obs": obs_t})
    if isinstance(out, dict):
        # Handle rare dict output cases
        for k in ("logits", "action_dist_inputs", "model_out"):
            if k in out:
                out = out[k]
                break
        else:
            raise RuntimeError(f"Unexpected dict keys from action_model: {list(out.keys())}")
    return out.detach().cpu().numpy()

def pure_out(obs_batch_np: np.ndarray) -> np.ndarray:
    obs_t = torch.from_numpy(obs_batch_np).float()
    return pure_actor(obs_t).detach().cpu().numpy()

def rllib_action(obs_np: np.ndarray, explore: bool):
    a, _state, _info = policy.compute_single_action(obs_np, explore=explore)
    return np.asarray(a, dtype=np.float32)

# -----------------------------
# Test 1: actor outputs match exactly
# -----------------------------
rng = np.random.default_rng(0)
B = 64
obs_batch = rng.standard_normal((B, OBS_DIM)).astype(np.float32)

o_rllib = rllib_actor_out(obs_batch)
o_pure = pure_out(obs_batch)

max_abs = float(np.max(np.abs(o_rllib - o_pure)))
print("\n[TEST 1] RLlib actor vs pure actor (6-dim)")
print("  shape:", o_rllib.shape)
print("  max |diff|:", max_abs)
print("  ✅ match" if max_abs < 1e-6 else "  ⚠️ mismatch")

# -----------------------------
# Test 2: deterministic action == mean
# -----------------------------
obs = rng.standard_normal((OBS_DIM,)).astype(np.float32)

raw = pure_out(obs[None, :])[0]  # (6,)
mean = raw[:ACT_DIM]
log_std = raw[ACT_DIM:]
std = np.exp(log_std)

a_det = rllib_action(obs, explore=False)
print("\n[TEST 2] Deterministic action vs mean")
print("  action_dim:", ACT_DIM)
print("  rllib det:", a_det)
print("  mean     :", mean)
print("  max|diff|:", float(np.max(np.abs(a_det - mean))))

# -----------------------------
# Test 3: stochastic action sampling sanity
# (Not exact equality because RLlib uses its own RNG, but it should look like mean+std*eps)
# -----------------------------
a_sto = rllib_action(obs, explore=True)
print("\n[TEST 3] Stochastic action sanity")
print("  rllib sto:", a_sto)
print("  mean     :", mean)
print("  std      :", std)
print("  z = (a-mean)/std :", (a_sto - mean) / (std + 1e-8))

# -----------------------------
# Minimal deployment helpers (pure numpy)
# -----------------------------
def deploy_action_deterministic(obs_np: np.ndarray) -> np.ndarray:
    raw = pure_out(obs_np[None, :])[0]
    return raw[:ACT_DIM].astype(np.float32)

def deploy_action_stochastic(obs_np: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    raw = pure_out(obs_np[None, :])[0]
    mean = raw[:ACT_DIM]
    log_std = raw[ACT_DIM:]
    std = np.exp(log_std)
    eps = rng.standard_normal((ACT_DIM,), dtype=np.float32)
    return (mean + std * eps).astype(np.float32)

print("\nDone.")
