"""
About: 
    Reward functions for GRPO training.
    Implements all three reward signals: binary_reward, toolrl_reward, irl_reward.
"""

import json
import re
from pathlib import Path
import numpy as np
from scipy import stats
import sys
from importlib.util import spec_from_file_location, module_from_spec as _mfs
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, OUTPUT_DIR,
    TOOLRL_W_OUTCOME, TOOLRL_W_NAME, TOOLRL_W_SCHEMA, TOOLRL_W_VALUES,
    IRL_REWARD_MIN, IRL_REWARD_MAX,
)

def binary_reward(traj: dict) -> float:
    # 1.0 if trajectory finished with give_answer (solved), 0.0 otherwise.
    return 1.0 if traj.get("finish_type") == "give_answer" else 0.0

def _check_tool_name_valid(step: dict) -> float:
    # Tool name is non-empty and not a hallucinated placeholder.
    name = step.get("tool_name") or ""
    if not name:
        return 0.0
    bad_patterns = ["tool_name", "function", "api_call", "placeholder", "example"]
    if any(p in name.lower() for p in bad_patterns):
        return 0.0
    return 1.0


def _check_args_schema(step: dict) -> float:
    args = step.get("tool_args") or {}
    if not isinstance(args, dict) or len(args) == 0:
        return 0.0
    return 1.0

def _check_arg_values(step: dict) -> float:
    args = step.get("tool_args") or {}
    if not isinstance(args, dict) or len(args) == 0:
        return 0.0
    non_empty = sum(1 for v in args.values() if v not in (None, "", [], {}))
    return non_empty / len(args)

def toolrl_reward(traj: dict) -> float:
    W_OUTCOME = TOOLRL_W_OUTCOME
    W_NAME    = TOOLRL_W_NAME
    W_SCHEMA  = TOOLRL_W_SCHEMA
    W_VALUES  = TOOLRL_W_VALUES

    tool_steps = [s for s in traj["steps"] if s["role"] == "assistant" and s["tool_name"]]

    if not tool_steps:
        return W_OUTCOME * binary_reward(traj)

    outcome = binary_reward(traj)
    name_score = np.mean([_check_tool_name_valid(s) for s in tool_steps])
    schema_score = np.mean([_check_args_schema(s) for s in tool_steps])
    value_score = np.mean([_check_arg_values(s) for s in tool_steps])

    return (
        W_OUTCOME * outcome
        + W_NAME * name_score
        + W_SCHEMA * schema_score
        + W_VALUES * value_score
    )

def _load_feature_module():
    spec = spec_from_file_location(
        "feature_extraction",
        Path(__file__).parent / "feature_extraction.py"
    )
    mod = _mfs(spec)
    spec.loader.exec_module(mod)
    return mod

_feat_mod = _load_feature_module()
compute_features = _feat_mod.compute_features


def load_theta() -> np.ndarray:
    for candidate in [Path("output/theta_weights.json"), Path("data/theta_weights.json")]:
        if candidate.exists():
            with open(candidate) as f:
                data = json.load(f)
            return np.array(data["theta"])
    raise FileNotFoundError("theta_weights.json not found in output/ or data/")


def irl_reward(traj: dict, theta: np.ndarray | None = None) -> float:
    """
    R(τ) = θᵀ φ(τ)
    Loads theta from disk if not provided.
    """
    if theta is None:
        theta = load_theta()
    features = np.array(compute_features(traj))
    return float(theta @ features)


def irl_reward_normalised(traj: dict, theta: np.ndarray | None = None,
                           r_min: float = IRL_REWARD_MIN, r_max: float = IRL_REWARD_MAX) -> float:
    """IRL reward clipped and normalised to [0, 1] for stable GRPO training."""
    raw = irl_reward(traj, theta)
    return (np.clip(raw, r_min, r_max) - r_min) / (r_max - r_min)



def reward_report():
    """
    Print reward distributions across expert trajectories for all three signals.
    Use this to verify rewards are meaningful before GRPO training.
    """
    with open(DATA_DIR / "expert_parsed.json") as f:
        expert = json.load(f)

    try:
        theta = load_theta()
        has_theta = True
    except FileNotFoundError:
        print("theta_weights.json not found — run 04_maxent_irl.py first")
        has_theta = False

    print("=== Reward Distributions on Expert Trajectories ===\n")

    for name, fn in [
        ("binary_reward", binary_reward),
        ("toolrl_reward", toolrl_reward),
    ]:
        scores = [fn(t) for t in expert]
        arr = np.array(scores)
        print(f"{name}:")
        print(f"  mean={arr.mean():.4f}  std={arr.std():.4f}  "
              f"min={arr.min():.4f}  max={arr.max():.4f}")
        hist = np.histogram(arr, bins=5, range=(0, 1))
        for count, edge in zip(hist[0], hist[1]):
            bar = "█" * (count // max(1, len(expert) // 40))
            print(f"  [{edge:.1f}-{edge+0.2:.1f}] {bar} {count}")
        print()

    if has_theta:
        raw_scores = [irl_reward(t, theta) for t in expert]
        norm_scores = [irl_reward_normalised(t, theta) for t in expert]
        arr_raw = np.array(raw_scores)
        arr_norm = np.array(norm_scores)
        print("irl_reward (raw θᵀφ):")
        print(f"  mean={arr_raw.mean():.4f}  std={arr_raw.std():.4f}  "
              f"min={arr_raw.min():.4f}  max={arr_raw.max():.4f}")
        print("irl_reward (normalised to [0,1]):")
        print(f"  mean={arr_norm.mean():.4f}  std={arr_norm.std():.4f}  "
              f"min={arr_norm.min():.4f}  max={arr_norm.max():.4f}")

    # Correlation between reward signals
    print("\nCorrelations between reward signals (on expert set):")


    b = np.array([binary_reward(t) for t in expert])
    h = np.array([toolrl_reward(t) for t in expert])
    if has_theta:
        irl = np.array([irl_reward_normalised(t, theta) for t in expert])
        rho_bh, _ = stats.spearmanr(b, h)
        rho_bi, _ = stats.spearmanr(b, irl)
        rho_hi, _ = stats.spearmanr(h, irl)
        print(f"  binary  vs toolrl : ρ = {rho_bh:.4f}")
        print(f"  binary  vs irl    : ρ = {rho_bi:.4f}")
        print(f"  toolrl  vs irl    : ρ = {rho_hi:.4f}")
        print("\n  (IRL should correlate with toolrl but not perfectly — "
              "if ρ≈1.0, IRL adds no new information)")
    else:
        rho_bh, _ = stats.spearmanr(b, h)
        print(f"  binary vs toolrl: ρ = {rho_bh:.4f}")


if __name__ == "__main__":
    reward_report()