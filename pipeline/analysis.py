"""
Post-training evaluation. 
"""

import json
import re
import argparse
from pathlib import Path
import numpy as np
from scipy import stats
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, OUTPUT_DIR, MODELS_DIR, BASE_MODEL, FEATURE_NAMES, QUALITY_FEATURE_IDX
from feature_extraction import compute_features, load_tool_schemas

OUTPUT_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """You are a tool-using AI assistant. Given a query, you must:
1. Decide which tool to call and what arguments to use.
2. Use the tool result to answer the query.
3. Provide a final answer.

Respond in this format:
[TOOL_CALL] tool_name({"arg1": "value1", "arg2": "value2"})
[TOOL_RESULT] (tool output will appear here)
[ANSWER] your final answer"""

CONDITIONS = ["sft", "binary", "toolrl", "irl"]

def parse_completion(query: str, completion: str) -> dict:
    steps = []
    idx = 0
    for m in re.finditer(r"\[TOOL_CALL\]\s+(\w+)\((.+?)\)(?=\s*\[|\s*$)", completion, re.DOTALL):
        name = m.group(1).strip()
        try:
            args = json.loads(m.group(2).strip())
            if not isinstance(args, dict):
                args = {}
        except Exception:
            args = {}
        steps.append({"step_idx": idx, "role": "assistant", "tool_name": name, "tool_args": args, "tool_output": None})
        idx += 1
    for m in re.finditer(r"\[TOOL_RESULT\]\s*(.*?)(?=\[TOOL_CALL\]|\[ANSWER\]|$)", completion, re.DOTALL):
        out = m.group(1).strip()
        if out:
            steps.append({"step_idx": idx, "role": "tool", "tool_name": None, "tool_args": None, "tool_output": out[:300]})
            idx += 1
    answer_m = re.search(r"\[ANSWER\]\s*(.*?)$", completion, re.DOTALL)
    final_answer = answer_m.group(1).strip() if answer_m else None
    tool_names = [s["tool_name"] for s in steps if s["role"] == "assistant" and s["tool_name"]]
    return {
        "id": "eval", "query": query, "domain": "", "api": "",
        "pass_rate": 0.0, "steps": steps, "final_answer": final_answer,
        "finish_type": "give_answer" if final_answer else "give_up_and_restart",
        "n_tool_calls": len(tool_names),
        "tool_names_used": list(dict.fromkeys(tool_names)),
        "api_list": [],
    }

def run_inference(model_dir: Path, queries: list[str], max_new_tokens: int = 512) -> list[str]:
    print(f"loading {model_dir.name}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="auto"
    )
    adapter_config = model_dir / "adapter_config.json"
    if adapter_config.exists():
        model = PeftModel.from_pretrained(base, str(model_dir))
        model = model.merge_and_unload()
    else:
        model = base

    model.eval()
    completions = []

    for i, query in enumerate(queries):
        prompt = f"{SYSTEM_PROMPT}\n\n[QUERY] {query}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(generated, skip_special_tokens=True)
        completions.append(completion)

        if (i + 1) % 10 == 0:
            print(f"...{i + 1}/{len(queries)} done")

    del model, base
    torch.cuda.empty_cache()
    return completions


def score_completions(queries: list[str], completions: list[str]) -> list[dict]:
    records = []
    for query, completion in zip(queries, completions):
        traj = parse_completion(query, completion)
        features = compute_features(traj)
        records.append({
            "finish_type": traj["finish_type"],
            "features": features,
            "n_tool_calls": traj["n_tool_calls"],
        })
    return records


def mann_whitney_h2(results_by_condition: dict) -> dict:
    """
    H2: IRL-GRPO vs SFT on implicit features, restricted to successful completions.
    One-sided Mann-Whitney U: IRL > SFT on each quality feature.
    """
    irl_success = [r for r in results_by_condition.get("irl", []) if r["finish_type"] == "give_answer"]
    sft_success = [r for r in results_by_condition.get("sft", []) if r["finish_type"] == "give_answer"]

    if len(irl_success) < 5 or len(sft_success) < 5:
        return {"note": "too few successful completions for Mann-Whitney U", "irl_success_n": len(irl_success), "sft_success_n": len(sft_success)}

    implicit_idx = list(range(2, 8))  # indices 2-7 are the implicit quality features
    test_results = {}

    for i in implicit_idx:
        name = FEATURE_NAMES[i]
        irl_vals = [r["features"][i] for r in irl_success]
        sft_vals = [r["features"][i] for r in sft_success]
        u, p = stats.mannwhitneyu(irl_vals, sft_vals, alternative="greater")
        test_results[name] = {
            "u_statistic": float(u),
            "p_value": float(p),
            "irl_mean": round(float(np.mean(irl_vals)), 4),
            "sft_mean": round(float(np.mean(sft_vals)), 4),
            "significant_p05": bool(p < 0.05),
        }

    return {
        "irl_success_n": len(irl_success),
        "sft_success_n": len(sft_success),
        "features": test_results,
        "note":"one-sided test: IRL > SFT on each implicit feature",
    }

def finish_type_distribution(records: list[dict]) -> dict:
    total = len(records)
    give_answer = sum(1 for r in records if r["finish_type"] == "give_answer")
    return {
        "total": total,
        "give_answer": give_answer,
        "give_up": total - give_answer,
        "success_rate": round(give_answer / max(total, 1), 4),
    }

def feature_means(records: list[dict]) -> dict:
    if not records:
        return {}
    feat_matrix = np.array([r["features"] for r in records])
    return {name: round(float(feat_matrix[:, i].mean()), 4)
            for i, name in enumerate(FEATURE_NAMES)}


def within_success_quality_analysis(records: list[dict]) -> dict:
    # Tests whether the IRL reward captures a real quality gradient within the successful set, independent of pass-rate label.
    theta_path = OUTPUT_DIR / "theta_weights.json"
    if not theta_path.exists():
        return {"note": "theta_weights.json not found"}

    with open(theta_path) as f:
        theta = np.array(json.load(f)["theta"])

    success = [r for r in records if r["finish_type"] == "give_answer"]
    if len(success) < 10:
        return {"note": f"too few successful completions ({len(success)}) for quartile split"}

    scores = np.array([theta @ np.array(r["features"]) for r in success])
    q25 = np.percentile(scores, 25)
    q75 = np.percentile(scores, 75)
    bottom = [r for r, s in zip(success, scores) if s <= q25]
    top = [r for r, s in zip(success, scores) if s >= q75]

    if len(top) < 3 or len(bottom) < 3:
        return {"note": "quartiles too small after split", "n_success": len(success)}

    implicit_idx = list(range(2, 8))
    test_results = {}
    for i in implicit_idx:
        name = FEATURE_NAMES[i]
        top_vals = [r["features"][i] for r in top]
        bot_vals = [r["features"][i] for r in bottom]
        u, p = stats.mannwhitneyu(top_vals, bot_vals, alternative="greater")
        test_results[name] = {
            "u_statistic": float(u),
            "p_value": float(p),
            "top_quartile_mean": round(float(np.mean(top_vals)), 4),
            "bottom_quartile_mean": round(float(np.mean(bot_vals)), 4),
            "significant_p05": bool(p < 0.05),
        }

    return {
        "n_success": len(success),
        "n_top_quartile": len(top),
        "n_bottom_quartile": len(bottom),
        "score_q25": round(float(q25), 4),
        "score_q75": round(float(q75), 4),
        "features": test_results,
        "note": "top vs bottom IRL-score quartile within successful completions only",
    }


def check_reward_hacking(condition: str, records: list[dict] | None = None) -> dict:
    """
    Two-part check:
    1. Training log: reward trend early vs late (scalar signal only).
    2. Feature-level: if records from inference are provided, compare
       call_success_rate and redundancy_avoidance between early and late
       training by using the cached completions split by position.
       arg_completeness and arg_correctness having negative theta weights
       means the agent may learn to omit arguments — detectable as
       call_success_rate dropping and redundancy rising in generated outputs.
    """
    log_path = MODELS_DIR / condition / "training_log.json"
    result = {}

    if log_path.exists():
        with open(log_path) as f:
            log = json.load(f)
        rewards = [e.get("reward") for e in log if e.get("reward") is not None]
        losses = [e.get("loss") for e in log if e.get("loss") is not None]

        if rewards:
            thirds = max(len(rewards) // 3, 1)
            early_r = rewards[:thirds]
            late_r = rewards[-thirds:]
            result["training_log"] = {
                "n_steps": len(rewards),
                "reward_early_mean": round(float(np.mean(early_r)), 4),
                "reward_late_mean": round(float(np.mean(late_r)), 4),
                "reward_trend": "increasing" if np.mean(late_r) > np.mean(early_r) else "flat_or_decreasing",
                "loss_final": round(float(losses[-1]), 4) if losses else None,
            }
        else:
            result["training_log"] = {"note": "no reward values in log"}
    else:
        result["training_log"] = {"note": f"no log at {log_path}"}

    # feature-level hacking check from generated trajectories
    # call_success_rate = feature index 5, redundancy_avoidance = index 3
    if records:
        call_success_idx = FEATURE_NAMES.index("call_success_rate")
        redundancy_idx = FEATURE_NAMES.index("redundancy_avoidance")
        arg_complete_idx = FEATURE_NAMES.index("arg_completeness")

        midpoint = len(records) // 2
        early_recs = records[:midpoint]
        late_recs = records[midpoint:]

        def mean_feat(recs, idx):
            vals = [r["features"][idx] for r in recs if r["features"]]
            return round(float(np.mean(vals)), 4) if vals else None

        result["feature_hacking_check"] = {
            "call_success_rate_early": mean_feat(early_recs, call_success_idx),
            "call_success_rate_late": mean_feat(late_recs, call_success_idx),
            "redundancy_avoidance_early": mean_feat(early_recs, redundancy_idx),
            "redundancy_avoidance_late": mean_feat(late_recs, redundancy_idx),
            "arg_completeness_early": mean_feat(early_recs, arg_complete_idx),
            "arg_completeness_late": mean_feat(late_recs, arg_complete_idx),
            "note": "drop in call_success_rate or arg_completeness late vs early signals exploitation of negative theta weights",
        }
    return result


def main(conditions: list[str], skip_inference: bool):
    load_tool_schemas()

    with open(DATA_DIR / "held_out_parsed.json") as f:
        held_out = json.load(f)
    queries = [t["query"] for t in held_out]
    print(f"held-out set: {len(queries)} trajectories")

    results_by_condition = {}

    for condition in conditions:
        cache_path = OUTPUT_DIR / f"eval_completions_{condition}.json"

        if skip_inference and cache_path.exists():
            print(f"loading cached completions for {condition}...")
            with open(cache_path) as f:
                completions = json.load(f)
        else:
            model_dir = MODELS_DIR / condition
            if not model_dir.exists():
                print(f"no model at {model_dir}, skipping {condition}")
                continue
            completions = run_inference(model_dir, queries)
            with open(cache_path, "w") as f:
                json.dump(completions, f, indent=2)

        records = score_completions(queries, completions)
        results_by_condition[condition] = records
        print(f"{condition}: {finish_type_distribution(records)}")

    # per-condition summary
    eval_results = {}
    for condition, records in results_by_condition.items():
        eval_results[condition] = {
            "finish_type": finish_type_distribution(records),
            "feature_means": feature_means(records),
            "feature_means_success_only": feature_means(
                [r for r in records if r["finish_type"] == "give_answer"]
            ),
        }

    with open(OUTPUT_DIR / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    print("saved output/eval_results.json")

    # IRL vs SFT on implicit features (successful completions)
    mw = mann_whitney_h2(results_by_condition)
    with open(OUTPUT_DIR / "mann_whitney_h2.json", "w") as f:
        json.dump(mw, f, indent=2)
    print("saved output/mann_whitney_h2.json")

    # within-success quality gradient: top vs bottom IRL-score quartile
    # run on IRL condition records; tests reward captures real gradient within successes
    within_success = {}
    for condition, records in results_by_condition.items():
        within_success[condition] = within_success_quality_analysis(records)
    with open(OUTPUT_DIR / "within_success_analysis.json", "w") as f:
        json.dump(within_success, f, indent=2)
    print("saved output/within_success_analysis.json")

    # reward hacking check: training log trend + feature-level exploitation check
    hacking_check = {}
    for condition, records in results_by_condition.items():
        # pass records only for IRL condition where negative weights make exploitation possible
        hacking_check[condition] = check_reward_hacking(
            condition,
            records=records if condition == "irl" else None
        )
    with open(OUTPUT_DIR / "reward_hacking_check.json", "w") as f:
        json.dump(hacking_check, f, indent=2)
    print("saved output/reward_hacking_check.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--skip-inference", action="store_true",
                        help="load cached completions if they exist, skip model loading")
    cli = parser.parse_args()
    main(cli.conditions, cli.skip_inference)
