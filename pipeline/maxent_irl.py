"""
Source Copyright:
    Ziebart, Brian D., et al. "Maximum entropy inverse reinforcement learning." AAAI. 2008.

About:
    Recovers reward weights θ from expert tool-use demonstrations.
    This is the "maximum likelihood IRL" approximation used when a policy rollout oracle is unavailable 
        (see also: Sun & van der Schaar arXiv 2405.15624; DR-IRL arXiv 2503.18991).
"""
import json
import random
from pathlib import Path
import numpy as np
from scipy import stats
import argparse

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, OUTPUT_DIR, MODELS_DIR, FEATURE_NAMES, N_FEATURES,
    IRL_LR, IRL_L2, IRL_ITERS, IRL_LOG_EVERY, IRL_PAIR_DELTA,)
OUTPUT_DIR.mkdir(exist_ok=True)

def load_suboptimal_features() -> np.ndarray:
    # Load feature vectors for suboptimal trajectories
    subopt_path = DATA_DIR / "suboptimal_features.json"
    if subopt_path.exists():
        with open(subopt_path) as f:
            data = json.load(f)
        features = np.array([r["features"] for r in data])
        print(f"Loaded {len(features)} suboptimal trajectories from {subopt_path}")
        return features

    # Fallback: use the lower-quality portion of held_out
    held_path = DATA_DIR / "held_out_features.json"
    if held_path.exists():
        with open(held_path) as f:
            data = json.load(f)
        subopt = [r for r in data if r.get("pass_rate", 1.0) < 0.8]
        if len(subopt) >= 20:
            features = np.array([r["features"] for r in subopt])
            print(f"Fallback: using {len(features)} held-out trajectories with pass_rate < 0.8")
            return features

    raise FileNotFoundError(
        "No suboptimal trajectory features found.\n"
        "Run Stage 1 with --save-suboptimal flag to collect failed ToolBench trajectories.\n"
        "These are real agent trajectories needed as the model distribution for MaxEnt IRL.\n"
    )

def softmax_probs(theta: np.ndarray, features: np.ndarray) -> np.ndarray:
    logits = features @ theta
    logits -= logits.max()
    probs = np.exp(logits)
    return probs / probs.sum()


def maxent_irl(
    expert_features: np.ndarray,
    model_features: np.ndarray,
    lr: float = IRL_LR,
    l2: float = IRL_L2,
    n_iters: int = IRL_ITERS,
) -> tuple[np.ndarray, list[dict]]:
    mu_E = expert_features.mean(axis=0)  # (k,) — fixed target

    # Pool: expert + suboptimal. The softmax is computed over this full set.
    all_features = np.vstack([expert_features, model_features]) 
    N_E = len(expert_features)

    theta = np.zeros(N_FEATURES)
    log = []

    for t in range(n_iters):
        probs = softmax_probs(theta, all_features)
        mu_theta = (probs[:, None] * all_features).sum(axis=0) 
        grad = mu_E - mu_theta - 2 * l2 * theta

        theta = theta + lr * grad
        expert_probs = softmax_probs(theta, all_features)[:N_E]
        log_lik = np.log(expert_probs + 1e-12).mean()
        grad_norm = np.linalg.norm(grad)

        if t % IRL_LOG_EVERY == 0 or t == n_iters - 1:
            mu_gap = np.linalg.norm(mu_E - mu_theta)
            print(f"  Iter {t:4d} | LogL={log_lik:.4f} | |grad|={grad_norm:.5f} | "
                  f"|μ_E - μ_θ|={mu_gap:.5f}")
            log.append({
                "iter": t,
                "log_likelihood": float(log_lik),
                "grad_norm": float(grad_norm),
                "mu_gap": float(mu_gap),
                "theta": theta.tolist(),
            })

    return theta, log

def pairwise_ranking_check(theta: np.ndarray) -> dict:
    with open(DATA_DIR / "held_out_features.json") as f:
        held_out = json.load(f)

    features = np.array([r["features"] for r in held_out])
    pass_rates = np.array([r["pass_rate"] for r in held_out])
    predicted = features @ theta

    #cross-quality ranking check (give_answer vs give_up)
    n_cross = n_correct_cross = 0
    for i in range(len(held_out)):
        for j in range(len(held_out)):
            if i == j:
                continue
            if pass_rates[i] > pass_rates[j] + IRL_PAIR_DELTA:
                n_cross += 1
                if predicted[i] > predicted[j]:
                    n_correct_cross += 1
    cross_acc = n_correct_cross / max(n_cross, 1)

    #within-success ranking check(non-trivial, more informative)
    successful_idx = [i for i in range(len(held_out)) if pass_rates[i] >= 1.0]
    efficiency_col = features[:, 7] if features.shape[1] > 7 else None
    within_pairs = within_correct = 0
    if len(successful_idx) >= 2 and efficiency_col is not None:
        for ai in successful_idx:
            for bi in successful_idx:
                if ai == bi:
                    continue
                if efficiency_col[ai] - efficiency_col[bi] > 0.15:
                    within_pairs += 1
                    if predicted[ai] > predicted[bi]:
                        within_correct += 1
    within_acc = within_correct / max(within_pairs, 1)

    #Spearman / Kendall over full held-out
    pass_rate_std = pass_rates.std()
    if pass_rate_std < 0.01:
        print("\theld-out pass_rates near-constant Spearman/Kendall corr uninformative.")
        tau, tau_p = float("nan"), float("nan")
        rho, rho_p = float("nan"), float("nan")
    else:
        tau, tau_p = stats.kendalltau(predicted, pass_rates)
        rho, rho_p = stats.spearmanr(predicted, pass_rates)

    print(f"\nSanity Check: Ranking")
    print(f"\tCross-quality (give_answer vs give_up): {n_cross} pairs")
    print(f"\tCross-quality accuracy: {cross_acc:.4f} (random=0.50, target≥0.65)")
    print(f"\tWithin-success pairs: {within_pairs}")
    print(f"\tWithin-success accuracy: {within_acc:.4f} (random=0.50, target≥0.60)")
    print(f"\tKendall= {tau if not np.isnan(tau) else 'N/A'}")
    print(f"\tSpearman= {rho if not np.isnan(rho) else 'N/A'}")

    return {
        "cross_quality_pairs": n_cross,
        "cross_quality_accuracy": float(cross_acc),
        "within_success_pairs": within_pairs,
        "within_success_accuracy": float(within_acc),
        "pair_delta": IRL_PAIR_DELTA,
        "kendall_tau": float(tau) if not np.isnan(tau) else None,
        "kendall_pval": float(tau_p) if not np.isnan(tau_p) else None,
        "spearman_rho": float(rho) if not np.isnan(rho) else None,
        "spearman_pval": float(rho_p) if not np.isnan(rho_p) else None,
        "pass_rate_std_held_out": float(pass_rate_std),
        "n_held_out": len(held_out),
    }


def main():
    with open(DATA_DIR / "expert_features.json") as f:
        expert_data = json.load(f)
    expert_features = np.array([r["features"] for r in expert_data])

    print(f"Expert trajectories: {len(expert_features)}")
    print(f"Feature dim: {N_FEATURES}")
    print(f"\nExpert feature expectations:")
    mu_E = expert_features.mean(axis=0)
    for name, val in zip(FEATURE_NAMES, mu_E):
        print(f"\t{name:<30}: {val:.4f}")

    # Model distribution: real suboptimal trajectories from ToolBench
    try:
        model_features = load_suboptimal_features()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return

    print(f"\nModel distribution : {len(model_features)} real suboptimal trajectories")
    mu_M = model_features.mean(axis=0)
    print(f"Suboptimal feature expectations:")
    for i, (name, val) in enumerate(zip(FEATURE_NAMES, mu_M)):
        print(f"\t{name:<30}: {val:.4f} (mu_E - mu_M = {mu_E[i] - val:+.4f})")

    print(f"\nRunning MaxEnt IRL ({IRL_ITERS} iters, lr={IRL_LR}, λ={IRL_L2}) ...")
    print(f"Approximating Z(θ) over {len(expert_features) + len(model_features)} trajectories")

    theta, training_log = maxent_irl(expert_features, model_features, LR, IRL_L2, IRL_ITERS)

    # Normalise for interpretability
    theta_softmax = np.exp(theta) / np.exp(theta).sum()

    print("\nRecovered Theta Weights")
    ranked = sorted(range(N_FEATURES), key=lambda i: theta[i], reverse=True)
    rank_of = {i: r + 1 for r, i in enumerate(ranked)}
    for i, name in enumerate(FEATURE_NAMES):
        tag = " (implicit)" if i >= 2 else ""
        print(f"\t{name}{tag}: raw={theta[i]:.4f}, norm={theta_softmax[i]:.4f}, rank={rank_of[i]}")

    implicit_theta = theta[2:]
    n_nonzero = int(np.sum(np.abs(implicit_theta) > 0.05))
    print(f"\nImplicit features with |θ| > 0.05: {n_nonzero} / 6")
    if n_nonzero >= 2:
        print("Hypothesis check PASSED")
    else:
        print("Hypothesis check FAILED — may indicate feature overlap or data issue")

    all_features = np.vstack([expert_features, model_features])
    probs_final = softmax_probs(theta, all_features)
    mu_theta_final = (probs_final[:, None] * all_features).sum(axis=0)
    gap = np.abs(mu_E - mu_theta_final)
    print(f"\nFeature matching quality:")
    for name, g in zip(FEATURE_NAMES, gap):
        bar = "█" * int(g * 40)
        print(f"\t{name:<30} {g:.4f}  {bar}")
    print(f"Mean gap: {gap.mean():.4f} (lower = better feature matching)")

    # Sanity check: pairwise ranking
    try:
        sanity = pairwise_ranking_check(theta)
    except FileNotFoundError:
        print("\n(held_out_features.json not found — run Stage 2 first)")
        sanity = {}

    # Save
    output = {
        "theta": theta.tolist(),
        "theta_normalised": theta_softmax.tolist(),
        "feature_names": FEATURE_NAMES,
        "hyperparameters": {"lr": IRL_LR, "IRL_L2": IRL_L2, "n_iters": IRL_ITERS,},
        "data": {
            "n_expert": len(expert_features),
            "n_model": len(model_features),
            "model_source": "real suboptimal ToolBench trajectories (pass_rate < 0.8)",
        },
        "expert_feature_expectations": mu_E.tolist(),
        "model_feature_expectations_final": mu_theta_final.tolist(),
        "feature_matching_gap": gap.tolist(),
        "hypothesis_check": {
            "n_implicit_nonzero": n_nonzero,
            "threshold": 0.05,
            "passed": bool(n_nonzero >= 2),
        },
        "sanity_check": sanity,
    }

    with open(OUTPUT_DIR / "theta_weights.json", "w") as f:
        json.dump(output, f, indent=2)
    with open(OUTPUT_DIR / "irl_training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)
    with open(OUTPUT_DIR / "sanity_check.json", "w") as f:
        json.dump(sanity, f, indent=2)

    print(f"\nSaved: output/theta_weights.json, irl_training_log.json, sanity_check.json")
    print(f"Done. Ready for Stage 4.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=IRL_LR)
    parser.add_argument("--l2", type=float, default=IRL_L2)
    parser.add_argument("--iters", type=int, default=IRL_ITERS)
    parser.add_argument("--pair-delta", type=float, default=IRL_PAIR_DELTA, help="Min pass_rate gap for a pair to count in ranking check")
    cli = parser.parse_args()
    IRL_LR = cli.lr
    IRL_L2 = cli.l2
    IRL_ITERS = cli.iters
    IRL_PAIR_DELTA = cli.pair_delta
    main()