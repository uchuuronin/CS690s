"""
Inspired by: Ziebart et al. "Maximum Entropy Inverse Reinforcement Learning." AAAI 2008.
"""
import json
import argparse
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.special import expit

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, OUTPUT_DIR,
    FEATURE_NAMES, N_FEATURES,
    IRL_LR, IRL_L2, IRL_ITERS, IRL_LOG_EVERY, IRL_PAIR_DELTA,
)

OUTPUT_DIR.mkdir(exist_ok=True)

def _load(split: str) -> list[dict]:
    path = DATA_DIR / f"{split}_features.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. run feature_extraction.py first")
    with open(path) as f:
        return json.load(f)

def load_suboptimal() -> tuple[np.ndarray, list[dict]]:
    try:
        data = _load("suboptimal")
    except FileNotFoundError:
        data = [r for r in _load("held_out") if r.get("pass_rate", 1.0) < 0.8]
        if len(data) < 20:
            raise FileNotFoundError("too few suboptimal trajectories")
        print(f"fallback: {len(data)} held-out with pass_rate<0.8")
    return np.array([r["features"] for r in data]), data

def build_dfsdt_pairs(expert_data: list[dict], subopt_data: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    # match expert to their query-sibling DFSDT branches by id
    expert_by_id = {r["id"]: np.array(r["features"]) for r in expert_data}
    phi_w, phi_l = [], []
    for r in subopt_data:
        base = r["id"].split("_branch_")[0] if "_branch_" in r["id"] else None
        if base and base in expert_by_id:
            phi_w.append(expert_by_id[base])
            phi_l.append(np.array(r["features"]))
    if not phi_w:
        return np.empty((0, N_FEATURES)), np.empty((0, N_FEATURES))
    return np.array(phi_w), np.array(phi_l)

def fit_scaler(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = features.mean(axis=0)
    sigma = features.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return mu, sigma

def standardise(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (X - mu) / sigma

def unstandardise_theta(theta_std: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return theta_std / sigma

def pool_importance_weights(n_expert: int, n_subopt: int) -> np.ndarray:
    n_total = n_expert + n_subopt
    w = np.empty(n_total)
    w[:n_expert] = (n_total / n_expert) / n_total
    w[n_expert:] = (n_total / n_subopt) / n_total
    w /= w.sum()
    return w

def _softmax_weighted(theta: np.ndarray, features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    logits = features @ theta
    logits -= logits.max()
    p = weights * np.exp(logits)
    return p / p.sum()

def maxent_irl(
    expert_std: np.ndarray,
    all_std: np.ndarray,
    is_weights: np.ndarray,
    lr: float,
    l2: float,
    n_iters: int,
) -> tuple[np.ndarray, list[dict]]:
    mu_E = expert_std.mean(axis=0)
    theta = np.zeros(N_FEATURES)
    log = []

    for t in range(n_iters):
        probs = _softmax_weighted(theta, all_std, is_weights)
        mu_theta = (probs[:, None] * all_std).sum(axis=0)
        grad = mu_E - mu_theta - 2 * l2 * theta
        theta = theta + lr * grad

        if t % IRL_LOG_EVERY == 0 or t == n_iters - 1:
            mu_gap = float(np.linalg.norm(mu_E - mu_theta))
            print(f"[maxent] iter={t:4d} |mu_E-mu_θ|={mu_gap:.5f} |grad|={np.linalg.norm(grad):.5f}")
            log.append({"iter": t, "mu_gap": mu_gap, "grad_norm": float(np.linalg.norm(grad)), "theta": theta.tolist()})
    return theta, log


def bt_irl(phi_w_std: np.ndarray,phi_l_std: np.ndarray,lr: float,l2: float,n_iters: int) -> tuple[np.ndarray, list[dict]]:
    delta = phi_w_std - phi_l_std
    theta = np.zeros(N_FEATURES)
    log = []

    for t in range(n_iters):
        margins = delta @ theta
        grad = (expit(-margins)[:, None] * delta).mean(axis=0) - 2 * l2 * theta
        theta = theta + lr * grad
        if t % IRL_LOG_EVERY == 0 or t == n_iters - 1:
            log_lik = float(np.mean(np.log(expit(margins) + 1e-12)))
            pair_acc = float(np.mean(margins > 0))
            print(f"[bt]     iter={t:4d} logL={log_lik:.4f} pair_acc={pair_acc:.4f} |grad|={np.linalg.norm(grad):.5f}")
            log.append({"iter": t, "log_likelihood": log_lik, "pair_accuracy": pair_acc, "grad_norm": float(np.linalg.norm(grad)), "theta": theta.tolist()})
    return theta, log


def pairwise_ranking_check(theta: np.ndarray, pair_delta: float) -> dict:
    held_out = _load("held_out")
    features = np.array([r["features"] for r in held_out])
    pass_rates = np.array([r["pass_rate"] for r in held_out])
    predicted = features @ theta

    n_cross = n_correct = 0
    for i in range(len(held_out)):
        for j in range(len(held_out)):
            if i == j:
                continue
            if pass_rates[i] > pass_rates[j] + pair_delta:
                n_cross += 1
                if predicted[i] > predicted[j]:
                    n_correct += 1
    cross_acc = n_correct / max(n_cross, 1)

    succ_idx = [i for i in range(len(held_out)) if pass_rates[i] >= 1.0]
    eff_col = features[:, 7] if features.shape[1] > 7 else None
    within_pairs = within_correct = 0
    if len(succ_idx) >= 2 and eff_col is not None:
        for ai in succ_idx:
            for bi in succ_idx:
                if ai == bi:
                    continue
                if eff_col[ai] - eff_col[bi] > 0.15:
                    within_pairs += 1
                    if predicted[ai] > predicted[bi]:
                        within_correct += 1
    within_acc = within_correct / max(within_pairs, 1)

    pass_rate_std = float(pass_rates.std())
    if pass_rate_std < 0.01:
        tau = tau_p = rho = rho_p = None
    else:
        tau, tau_p = stats.kendalltau(predicted, pass_rates)
        rho, rho_p = stats.spearmanr(predicted, pass_rates)
        tau, tau_p, rho, rho_p = float(tau), float(tau_p), float(rho), float(rho_p)

    print(f"cross_quality: {n_cross} pairs, acc={cross_acc:.4f} (random=0.50, target≥0.65)")
    print(f"within_success: {within_pairs} pairs, acc={within_acc:.4f}")
    print(f"kendall_tau={tau}  spearman_rho={rho}")

    return {
        "cross_quality_pairs": n_cross, "cross_quality_accuracy": float(cross_acc),
        "within_success_pairs": within_pairs, "within_success_accuracy": float(within_acc),
        "pair_delta": pair_delta,
        "kendall_tau": tau, "kendall_pval": tau_p,
        "spearman_rho": rho, "spearman_pval": rho_p,
        "pass_rate_std_held_out": pass_rate_std, "n_held_out": len(held_out),
    }


def _save(theta_orig, training_log, sanity, n_expert, n_data, all_features_orig, is_weights, mu_E_orig, method, hyperparams, mu, sigma) -> dict:
    ranked = sorted(range(N_FEATURES), key=lambda i: theta_orig[i], reverse=True)
    n_nonzero = int(np.sum(np.abs(theta_orig[2:8]) > 0.05))
    print(f"[{method}] implicit |theta|>0.05: {n_nonzero}/6. H1 {'PASSED' if n_nonzero >= 2 else 'FAILED'}")

    gap = mean_gap = None
    if all_features_orig is not None and is_weights is not None:
        probs = _softmax_weighted(theta_orig, all_features_orig, is_weights)
        mu_theta = (probs[:, None] * all_features_orig).sum(axis=0)
        gap = np.abs(mu_E_orig - mu_theta).tolist()
        mean_gap = float(np.mean(np.abs(mu_E_orig - mu_theta)))

    output = {
        "method": method,
        "theta": theta_orig.tolist(),
        "theta_normalised": (np.exp(theta_orig) / np.exp(theta_orig).sum()).tolist(),
        "feature_names": FEATURE_NAMES,
        "feature_ranking": [FEATURE_NAMES[i] for i in ranked],
        "hyperparameters": hyperparams,
        "standardisation": {"feature_mean": mu.tolist(), "feature_std": sigma.tolist()},
        "data": {"n_expert": n_expert, "n_data": n_data},
        "expert_feature_expectations": mu_E_orig.tolist(),
        "feature_matching_gap": gap,
        "mean_feature_gap": mean_gap,
        "hypothesis_check": {"n_implicit_nonzero": n_nonzero, "threshold": 0.05, "passed": bool(n_nonzero >= 2)},
        "sanity_check": sanity,
    }

    with open(OUTPUT_DIR / f"theta_{method}.json", "w") as f:
        json.dump(output, f, indent=2)
    with open(OUTPUT_DIR / f"irl_training_log_{method}.json", "w") as f:
        json.dump(training_log, f, indent=2)
    return output

def main(lr, l2, n_iters, pair_delta, reward_source):
    expert_data = _load("expert")
    expert_orig = np.array([r["features"] for r in expert_data])
    model_orig, subopt_data = load_suboptimal()
    all_orig = np.vstack([expert_orig, model_orig])

    print(f"expert={len(expert_data)} suboptimal={len(subopt_data)} features={N_FEATURES}")

    # scaler on full pool: consistent across both objectives and GRPO
    mu, sigma = fit_scaler(all_orig)
    expert_std = standardise(expert_orig, mu, sigma)
    all_std = standardise(all_orig, mu, sigma)
    is_weights = pool_importance_weights(len(expert_data), len(subopt_data))
    mu_E_orig = expert_orig.mean(axis=0)

    print(f"\nMaxEnt IRL ({n_iters} iters, IS-weighted)...")
    theta_me_std, log_me = maxent_irl(expert_std, all_std, is_weights, lr, l2, n_iters)
    theta_me = unstandardise_theta(theta_me_std, sigma)
    try:
        sanity_me = pairwise_ranking_check(theta_me, pair_delta)
    except FileNotFoundError:
        sanity_me = {}
    out_me = _save(theta_me, log_me, sanity_me, len(expert_data), len(subopt_data), all_orig,is_weights,
                   mu_E_orig, "maxent", {"lr": lr, "l2": l2, "n_iters": n_iters, "is_weighted": True}, mu, sigma)

    phi_w_orig, phi_l_orig = build_dfsdt_pairs(expert_data, subopt_data)
    n_pairs = len(phi_w_orig)
    print(f"\nBT IRL ({n_pairs} query-matched pairs, {n_iters} iters)...")

    if n_pairs == 0:
        print("WARNING: no query-matched pairs; check suboptimal id format (_branch_ suffix)")
        out_bt = {**out_me, "method": "bt_skipped"}
    else:
        phi_w_std = standardise(phi_w_orig, mu, sigma)
        phi_l_std = standardise(phi_l_orig, mu, sigma)
        theta_bt_std, log_bt = bt_irl(phi_w_std, phi_l_std, lr, l2, n_iters)
        theta_bt = unstandardise_theta(theta_bt_std, sigma)
        try:
            sanity_bt = pairwise_ranking_check(theta_bt, pair_delta)
        except FileNotFoundError:
            sanity_bt = {}
        out_bt = _save(theta_bt, log_bt, sanity_bt, len(expert_data), n_pairs, None, None, mu_E_orig, 
                       "bt", {"lr": lr, "l2": l2, "n_iters": n_iters, "n_pairs": n_pairs}, mu, sigma)

    corr, _ = stats.spearmanr(theta_me, np.array(out_bt["theta"]))
    with open(OUTPUT_DIR / "theta_comparison.json", "w") as f:
        json.dump({
            "spearman_rho_maxent_vs_bt": float(corr),
            "maxent_ranking": out_me["feature_ranking"],
            "bt_ranking": out_bt["feature_ranking"],
        }, f, indent=2)
    print(f"\nθ agreement (MaxEnt vs BT): Spearman ρ={corr:.4f}")

    chosen = out_me if reward_source == "maxent" else out_bt
    with open(OUTPUT_DIR / "theta_weights.json", "w") as f:
        json.dump(chosen, f, indent=2)
    with open(OUTPUT_DIR / "sanity_check.json", "w") as f:
        json.dump(chosen["sanity_check"], f, indent=2)
    print("saved: theta_maxent.json, theta_bt.json, theta_comparison.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=IRL_LR)
    parser.add_argument("--l2", type=float, default=IRL_L2)
    parser.add_argument("--iters", type=int, default=IRL_ITERS)
    parser.add_argument("--pair-delta", type=float, default=IRL_PAIR_DELTA)
    parser.add_argument("--reward-source", choices=["maxent", "bt"], default="maxent")
    cli = parser.parse_args()
    main(cli.lr, cli.l2, cli.iters, cli.pair_delta, cli.reward_source)