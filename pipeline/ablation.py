"""
Three analyses
  1. Leave-one-out feature ablation 
  2. BT vs MaxEnt theta comparison.
  3. Decoy stability check.
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, OUTPUT_DIR,
    FEATURE_NAMES, N_FEATURES, QUALITY_FEATURE_IDX, DECOY_FEATURE_IDX,
    IRL_LR, IRL_L2, IRL_ITERS, IRL_PAIR_DELTA,
)
from maxent_irl import (
    _load, load_suboptimal, fit_scaler, standardise, unstandardise_theta,
    pool_importance_weights, _softmax_weighted, maxent_irl, pairwise_ranking_check,
)

OUTPUT_DIR.mkdir(exist_ok=True)


def run_irl_subset(expert_feat: np.ndarray, model_feat: np.ndarray,
                   active_cols: list[int]) -> np.ndarray:
    """Run MaxEnt IRL on a column subset. Returns theta in original (full) feature space."""
    exp_sub = expert_feat[:, active_cols]
    mod_sub = model_feat[:, active_cols]
    all_sub = np.vstack([exp_sub, mod_sub])

    mu = all_sub.mean(axis=0)
    sigma = all_sub.std(axis=0)
    sigma[sigma < 1e-8] = 1.0

    exp_std = (exp_sub - mu) / sigma
    all_std = (all_sub - mu) / sigma

    is_w = pool_importance_weights(len(expert_feat), len(model_feat))
    n_sub = len(active_cols)

    # minimal gradient ascent on subset features
    mu_E = exp_std.mean(axis=0)
    theta_sub = np.zeros(n_sub)
    for _ in range(IRL_ITERS):
        probs = _softmax_weighted(theta_sub, all_std, is_w)
        mu_theta = (probs[:, None] * all_std).sum(axis=0)
        grad = mu_E - mu_theta - 2 * IRL_L2 * theta_sub
        theta_sub = theta_sub + IRL_LR * grad

    # map back to original scale
    theta_orig_sub = theta_sub / sigma

    # embed into full N_FEATURES vector (zeros for dropped features)
    theta_full = np.zeros(N_FEATURES)
    for out_idx, orig_idx in enumerate(active_cols):
        theta_full[orig_idx] = theta_orig_sub[out_idx]

    return theta_full


def ranking_accuracy(theta: np.ndarray, pair_delta: float = IRL_PAIR_DELTA) -> float:
    held_out = _load("held_out")
    features = np.array([r["features"] for r in held_out])
    pass_rates = np.array([r["pass_rate"] for r in held_out])
    predicted = features @ theta

    n = n_correct = 0
    for i in range(len(held_out)):
        for j in range(len(held_out)):
            if i == j:
                continue
            if pass_rates[i] > pass_rates[j] + pair_delta:
                n += 1
                if predicted[i] > predicted[j]:
                    n_correct += 1
    return n_correct / max(n, 1)


def leave_one_out_ablation(expert_feat: np.ndarray, model_feat: np.ndarray) -> dict:
    print("leave-one-out ablation (8 runs)...")

    # baseline: all quality features
    baseline_acc = ranking_accuracy(
        run_irl_subset(expert_feat, model_feat, QUALITY_FEATURE_IDX)
    )
    print(f"baseline (all 8 quality features): acc={baseline_acc:.4f}")

    results = {"baseline_accuracy": float(baseline_acc), "dropped": {}}

    for drop_idx in QUALITY_FEATURE_IDX:
        active = [i for i in QUALITY_FEATURE_IDX if i != drop_idx]
        theta = run_irl_subset(expert_feat, model_feat, active)
        acc = ranking_accuracy(theta)
        drop = baseline_acc - acc
        name = FEATURE_NAMES[drop_idx]
        results["dropped"][name] = {
            "accuracy_without": round(float(acc), 4),
            "accuracy_drop": round(float(drop), 4),
            "load_bearing": bool(drop > 0.03),  # >3pp drop means this feature matters
        }
        print(f"drop {name}: acc={acc:.4f} (drop={drop:+.4f})")

    return results


def decoy_stability_check(expert_feat: np.ndarray, model_feat: np.ndarray) -> dict:
    print("decoy stability check...")

    theta_full = run_irl_subset(expert_feat, model_feat, list(range(N_FEATURES)))
    theta_no_decoy = run_irl_subset(expert_feat, model_feat, QUALITY_FEATURE_IDX)

    # compare quality feature weights only
    q_full = np.array([theta_full[i] for i in QUALITY_FEATURE_IDX])
    q_nodecoy = np.array([theta_no_decoy[i] for i in QUALITY_FEATURE_IDX])

    corr, _ = stats.spearmanr(q_full, q_nodecoy)
    max_shift = float(np.max(np.abs(q_full - q_nodecoy)))
    mean_shift = float(np.mean(np.abs(q_full - q_nodecoy)))

    per_feature = {}
    for i, idx in enumerate(QUALITY_FEATURE_IDX):
        name = FEATURE_NAMES[idx]
        per_feature[name] = {
            "theta_with_decoys": round(float(q_full[i]), 4),
            "theta_without_decoys": round(float(q_nodecoy[i]), 4),
            "shift": round(float(q_full[i] - q_nodecoy[i]), 4),
        }

    print(f"Spearman rho between theta_with vs theta_without decoys: {corr:.4f}")
    print(f"max shift: {max_shift:.4f}, mean shift: {mean_shift:.4f}")

    return {
        "spearman_rho": float(corr),
        "max_shift": round(max_shift, 4),
        "mean_shift": round(mean_shift, 4),
        "high_collinearity_with_decoys": bool(max_shift > 0.1),
        "per_feature": per_feature,
    }


def interpret_theta_comparison() -> dict:
    comp_path = OUTPUT_DIR / "theta_comparison.json"
    if not comp_path.exists():
        return {"note": "theta_comparison.json not found. run maxent_irl.py first"}

    with open(comp_path) as f:
        comp = json.load(f)

    maxent_rank = {name: i for i, name in enumerate(comp["maxent_ranking"])}
    bt_rank = {name: i for i, name in enumerate(comp["bt_ranking"])}

    disagreements = []
    for name in FEATURE_NAMES:
        if name not in maxent_rank or name not in bt_rank:
            continue
        rank_diff = abs(maxent_rank[name] - bt_rank[name])
        if rank_diff >= 3:
            disagreements.append({
                "feature": name,
                "maxent_rank": maxent_rank[name] + 1,
                "bt_rank": bt_rank[name] + 1,
                "rank_diff": rank_diff,
            })

    disagreements.sort(key=lambda x: x["rank_diff"], reverse=True)

    return {
        "spearman_rho_theta_vectors": comp.get("spearman_rho_maxent_vs_bt"),
        "features_with_large_rank_diff": disagreements,
        "interpretation": (
            "objectives agree" if comp.get("spearman_rho_maxent_vs_bt", 0) > 0.8
            else "objectives disagree on feature importance"
        ),
    }


def main():
    expert_data = _load("expert")
    expert_feat = np.array([r["features"] for r in expert_data])
    model_feat, _ = load_suboptimal()

    print(f"expert={len(expert_feat)} model={len(model_feat)}")

    loo = leave_one_out_ablation(expert_feat, model_feat)
    with open(OUTPUT_DIR / "ablation_loo.json", "w") as f:
        json.dump(loo, f, indent=2)
    print("saved output/ablation_loo.json")

    decoy = decoy_stability_check(expert_feat, model_feat)
    with open(OUTPUT_DIR / "ablation_decoy.json", "w") as f:
        json.dump(decoy, f, indent=2)
    print("saved output/ablation_decoy.json")

    interp = interpret_theta_comparison()
    with open(OUTPUT_DIR / "theta_comparison_interp.json", "w") as f:
        json.dump(interp, f, indent=2)
    print("saved output/theta_comparison_interp.json")


if __name__ == "__main__":
    main()
