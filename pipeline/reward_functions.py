"""
Inspired by: Qian et al. "ToolRL: Reward is All Tool Learning Needs." NeurIPS 2025.

Reward functions for GRPO training: binary, ToolRL-style, and IRL-recovered.
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, OUTPUT_DIR, TOOLRL_WEIGHTS, IRL_REWARD_CLIP
from feature_extraction import compute_features

def binary_reward(traj: dict) -> float:
    return 1.0 if traj.get("finish_type") == "give_answer" else 0.0

def _check_tool_name_valid(step: dict) -> float:
    name = step.get("tool_name") or ""
    if not name:
        return 0.0
    if any(p in name.lower() for p in ["tool_name", "function", "api_call", "placeholder", "example"]):
        return 0.0
    return 1.0

def _check_args_schema(step: dict) -> float:
    args = step.get("tool_args") or {}
    return 1.0 if isinstance(args, dict) and len(args) > 0 else 0.0

def _check_arg_values(step: dict) -> float:
    args = step.get("tool_args") or {}
    if not isinstance(args, dict) or len(args) == 0:
        return 0.0
    non_empty = sum(1 for v in args.values() if v not in (None, "", [], {}))
    return non_empty/len(args)

def toolrl_reward(traj: dict) -> float:
    # Decomposed reward: outcome + tool name validity + argument schema + argument values.
    w = TOOLRL_WEIGHTS
    tool_steps = [s for s in traj["steps"] if s["role"] == "assistant" and s["tool_name"]]
    if not tool_steps:
        return w["outcome"] * binary_reward(traj)
    return (
        w["outcome"] * binary_reward(traj)
        + w["name"] * float(np.mean([_check_tool_name_valid(s) for s in tool_steps]))
        + w["schema"] * float(np.mean([_check_args_schema(s) for s in tool_steps]))
        + w["values"] * float(np.mean([_check_arg_values(s) for s in tool_steps]))
    )

def load_theta() -> np.ndarray:
    for candidate in [OUTPUT_DIR / "theta_weights.json", DATA_DIR / "theta_weights.json"]:
        if candidate.exists():
            with open(candidate) as f:
                return np.array(json.load(f)["theta"])
    raise FileNotFoundError("theta_weights.json not found; run maxent_irl.py first")

def irl_reward(traj: dict, theta: np.ndarray | None = None) -> float:
    if theta is None:
        theta = load_theta()
    return float(theta @ np.array(compute_features(traj)))

def irl_reward_normalised(traj: dict, theta: np.ndarray | None = None) -> float:
    r_min, r_max = IRL_REWARD_CLIP
    raw = irl_reward(traj, theta)
    return (np.clip(raw, r_min, r_max) - r_min) / (r_max - r_min)


def reward_correlations(expert: list[dict], theta: np.ndarray) -> dict:
    # Returns Spearman correlations between the three reward signals on the expert set.
    b = np.array([binary_reward(t) for t in expert])
    h = np.array([toolrl_reward(t) for t in expert])
    irl = np.array([irl_reward_normalised(t, theta) for t in expert])
    return {
        "binary_vs_toolrl": float(stats.spearmanr(b, h).statistic),
        "binary_vs_irl": float(stats.spearmanr(b, irl).statistic),
        "toolrl_vs_irl": float(stats.spearmanr(h, irl).statistic),
    }

def stats_for(arr):
        return {"mean": round(float(arr.mean()), 4), "std": round(float(arr.std()), 4), "min": round(float(arr.min()), 4), "max": round(float(arr.max()), 4)}

def compute_reward_stats(expert: list[dict]) -> dict:
    # Saves reward distribution stats and inter-signal correlations to output/reward_stats.json.
    theta = load_theta()

    b = np.array([binary_reward(t) for t in expert])
    h = np.array([toolrl_reward(t) for t in expert])
    irl_raw = np.array([irl_reward(t, theta) for t in expert])
    irl_norm = np.array([irl_reward_normalised(t, theta) for t in expert])
    result = {
        "binary": stats_for(b),
        "toolrl": stats_for(h),
        "irl_raw": stats_for(irl_raw),
        "irl_normalised": stats_for(irl_norm),
        "correlations": reward_correlations(expert, theta),
    }
    with open(OUTPUT_DIR / "reward_stats.json", "w") as f:
        json.dump(result, f, indent=2)
    return result

if __name__ == "__main__":
    with open(DATA_DIR / "expert_parsed.json") as f:
        expert = json.load(f)
    stats = compute_reward_stats(expert)
    print(f"reward stats saved to {OUTPUT_DIR}/reward_stats.json")
    print(f"toolrl_vs_irl = {stats['correlations']['toolrl_vs_irl']:.4f}")
