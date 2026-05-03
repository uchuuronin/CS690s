"""
Computes an 11-dimensional feature vector φ(τ) per trajectory (8 quality + 3 decoy).
All features normalised to [0, 1].
"""

import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, OUTPUT_DIR, FEATURE_NAMES, N_FEATURES, QUALITY_FEATURE_IDX, DECOY_FEATURE_IDX

OUTPUT_DIR.mkdir(exist_ok=True)

_TOOL_SCHEMA: dict[str, dict] = {}
_SCHEMA_LOADED = False

INFO_TOOL_KEYWORDS = {
    "search", "get", "fetch", "retrieve", "query", "lookup", "find",
    "list", "info", "detail", "check", "read", "load",
}
ACTION_TOOL_KEYWORDS = {
    "post", "create", "update", "delete", "send", "submit", "write",
    "book", "buy", "pay", "set", "modify", "cancel",
}
CONSTRAINT_KEYWORDS = [
    "free", "cheap", "budget", "no", "without", "exclude", "avoid",
    "only", "must", "require", "limit", "max", "min",
]

_ERROR_PATTERNS = [
    '"error":', "traceback", "404", "403", "500",
    "rate limit", "invalid api", "unauthorized",
]
_PSEUDO_TOOLS = {"finish", "finishaction"}


def load_tool_schemas(toolenv_path: Path | None = None) -> dict:
    global _TOOL_SCHEMA, _SCHEMA_LOADED
    if toolenv_path is None:
        toolenv_path = DATA_DIR / "toolenv" / "tools"
    if not toolenv_path.exists():
        _SCHEMA_LOADED = True
        return _TOOL_SCHEMA

    n_loaded = 0
    for json_path in toolenv_path.rglob("*.json"):
        try:
            with open(json_path) as f:
                raw = json.load(f)
        except Exception:
            continue
        std_name = raw.get("standardized_name", "").lower().strip()
        tool_name = raw.get("tool_name", "").lower().strip()
        description = raw.get("tool_description", "") or raw.get("description", "")
        api_index = {}
        for api in raw.get("api_list", []):
            api_name = (api.get("name") or "").lower().strip()
            if not api_name:
                continue
            api_index[api_name] = {
                "required_params": [p.get("name", "").lower() for p in api.get("required_parameters", []) if isinstance(p, dict)],
                "optional_params": [p.get("name", "").lower() for p in api.get("optional_parameters", []) if isinstance(p, dict)],
                "method": (api.get("method") or "GET").upper(),
                "description": api.get("description", ""),
            }
        entry = {"tool_name": tool_name, "description": description, "api_list": api_index}
        if std_name:
            _TOOL_SCHEMA[std_name] = entry
        if tool_name and tool_name != std_name:
            _TOOL_SCHEMA[tool_name] = entry
        n_loaded += 1

    _SCHEMA_LOADED = True
    return _TOOL_SCHEMA


def _get_tool_entry(tool_name: str) -> dict | None:
    if not tool_name:
        return None
    key = tool_name.lower().strip()
    if key in _TOOL_SCHEMA:
        return _TOOL_SCHEMA[key]
    for variant in [key.replace("_", ""), key.split("_")[0], key.split(".")[0]]:
        if variant in _TOOL_SCHEMA:
            return _TOOL_SCHEMA[variant]
    return None


def _get_api_entry(tool_name: str, api_name: str) -> dict | None:
    entry = _get_tool_entry(tool_name)
    if not entry:
        return None
    api_key = (api_name or "").lower().strip()
    api_list = entry.get("api_list", {})
    if api_key in api_list:
        return api_list[api_key]
    for k, v in api_list.items():
        if api_key in k or k in api_key:
            return v
    if len(api_list) == 1:
        return next(iter(api_list.values()))
    return None


def _parse_tool_output(output: str) -> dict:
    """
    Parse a tool response into its meaningful components rather than brute truncating.
    Returns a dict with: is_success, has_data, response_text, error_msg, n_results.
    """
    if not output or not output.strip():
        return {"is_success": False, "has_data": False, "response_text": "", "error_msg": "empty", "n_results": 0}

    low = output.lower().strip()

    if low in ("{}", "[]", '""', "null", "none", ""):
        return {"is_success": False, "has_data": False, "response_text": "", "error_msg": "empty_body", "n_results": 0}

    is_error = any(p in low for p in _ERROR_PATTERNS)
    response_text = ""
    error_msg = ""
    n_results = 0

    try:
        parsed = json.loads(output)
        if isinstance(parsed, dict):
            response_text = str(parsed.get("response", "") or parsed.get("result", "") or parsed.get("data", ""))
            error_msg = str(parsed.get("error", "") or parsed.get("message", "") if is_error else "")
            n_results = len(parsed.get("results", parsed.get("items", parsed.get("data", [])) if isinstance(parsed.get("data"), list) else []))
        elif isinstance(parsed, list):
            n_results = len(parsed)
            response_text = json.dumps(parsed[:3])  # first 3 items for context
    except Exception:
        # Not JSON — use raw text, cap at meaningful length
        response_text = output[:300] if not is_error else ""
        error_msg = output[:100] if is_error else ""

    has_data = bool(response_text.strip()) or n_results > 0
    if is_error and not has_data:
        is_success = False
    elif is_error and has_data:
        is_success = bool(re.search(r'"response"\s*:\s*"([^"]{10,})"', output))
    else:
        is_success = has_data or len(output.strip()) > 5

    return {
        "is_success": is_success,
        "has_data": has_data,
        "response_text": response_text,
        "error_msg": error_msg,
        "n_results": n_results,
    }


def f0_tool_selection_accuracy(traj: dict) -> float:
    intended = {
        (item.get("tool_name") or "").lower()
        for item in traj.get("api_list", [])
        if isinstance(item, dict)
    }
    called = [t.lower() for t in traj["tool_names_used"]
               if t and t.lower() not in _PSEUDO_TOOLS]
    if not called:
        return 0.0
    if intended:
        matches = sum(1 for c in called if any(i in c or c in i for i in intended))
        schema_score = matches / len(called)
        query_words = set(re.findall(r"\b\w{4,}\b", traj["query"].lower()))
        desc_hits = sum(
            1 for tool in called
            if (e := _get_tool_entry(tool)) and
               query_words & set(re.findall(r"\b\w{4,}\b", e["description"].lower()))
        )
        return min(1.0, schema_score * 0.85 + 0.15 * (desc_hits / len(called)))
    else:
        query_words = set(re.findall(r"\b\w{4,}\b", traj["query"].lower()))
        hits = sum(1 for t in called if query_words & set(re.split(r"[_\-\s]", t)))
        solved_bonus = 0.2 if traj["finish_type"] == "give_answer" else 0.0
        return min(1.0, (hits / len(called)) * 0.8 + solved_bonus)


def f1_arg_correctness(traj: dict) -> float:
    tool_steps = [s for s in traj["steps"] if s["role"] == "assistant" and s["tool_name"]]
    if not tool_steps:
        return 0.0
    scores = []
    for s in tool_steps:
        args = s.get("tool_args") or {}
        if not isinstance(args, dict):
            scores.append(0.0)
            continue
        api_entry = _get_api_entry(s["tool_name"], s["tool_name"])
        if api_entry is None:
            entry = _get_tool_entry(s["tool_name"])
            if entry and len(entry["api_list"]) == 1:
                api_entry = next(iter(entry["api_list"].values()))
        if api_entry:
            required = api_entry["required_params"]
            args_lower = {k.lower() for k in args}
            req_score = (sum(1 for r in required if r in args_lower) / len(required)) if required else (1.0 if args else 0.5)
            optional = api_entry["optional_params"]
            opt_bonus = 0.1 * (sum(1 for o in optional if o in args_lower) / len(optional)) if optional else 0.0
            scores.append(min(1.0, req_score + opt_bonus))
        else:
            has_value = any(v not in (None, "", [], {}) for v in args.values())
            scores.append(1.0 if (args and has_value) else 0.0)
    return float(np.mean(scores))


def f2_tool_diversity(traj: dict) -> float:
    """
    Fraction of distinct tool names used, with penalty for long consecutive same-tool runs.
    Replaces schema-dependent call_ordering (near-constant without toolenv access).
    """
    tool_steps = [s for s in traj["steps"]
                  if s["role"] == "assistant" and s["tool_name"]
                  and s["tool_name"].lower() not in _PSEUDO_TOOLS]
    if not tool_steps:
        return 0.5
    names = [s["tool_name"].lower() for s in tool_steps]
    diversity = len(set(names)) / len(names)
    max_run = cur_run = 1
    for i in range(1, len(names)):
        cur_run = cur_run + 1 if names[i] == names[i - 1] else 1
        max_run = max(max_run, cur_run)
    loop_penalty = max(0.0, (max_run - 3) * 0.1)
    return max(0.0, diversity - loop_penalty)


def f3_redundancy_avoidance(traj: dict) -> float:
    tool_steps = [s for s in traj["steps"]
                  if s["role"] == "assistant" and s["tool_name"]
                  and s["tool_name"].lower() not in _PSEUDO_TOOLS]
    if not tool_steps:
        return 1.0
    names = [s["tool_name"].lower() for s in tool_steps]
    consecutive = sum(1 for i in range(1, len(names)) if names[i] == names[i - 1])
    return 1.0 - consecutive / max(len(names) - 1, 1)


def f4_constraint_adherence(traj: dict) -> float:
    query_lower = traj["query"].lower()
    active = [kw for kw in CONSTRAINT_KEYWORDS if kw in query_lower]
    if not active:
        return 1.0
    arg_blob = " ".join(
        " ".join(str(v) for v in (s.get("tool_args") or {}).values()).lower()
        for s in traj["steps"]
        if isinstance(s.get("tool_args"), dict)
    )
    return sum(1 for kw in active if kw in arg_blob) / len(active)


def f5_call_success_rate(traj: dict) -> float:
    """
    Fraction of tool calls with a successful, non-empty response.
    Replaces info_sufficiency (ρ=−0.22 inverted correlation): failed DFSDT
    trajectories accumulate more calls, inflating keyword-based sufficiency scores.
    """
    result_steps = [s for s in traj["steps"]
                    if s["role"] in ("tool", "function", "observation")
                    and s.get("tool_output") is not None]
    if not result_steps:
        return 0.0
    successes = sum(
        1 for s in result_steps
        if _parse_tool_output(s.get("tool_output", "") or "")["is_success"]
    )
    return successes / len(result_steps)


def f6_arg_completeness(traj: dict) -> float:
    """
    Fraction of tool calls with at least one non-trivial argument value.
    Replaces invasiveness_minimisation (mean≈0.995, near-zero variance on benign ToolBench tasks).
    """
    tool_steps = [s for s in traj["steps"]
                  if s["role"] == "assistant" and s["tool_name"]
                  and s["tool_name"].lower() not in _PSEUDO_TOOLS]
    if not tool_steps:
        return 0.0
    complete = sum(
        1 for s in tool_steps
        if isinstance(s.get("tool_args"), dict)
        and any(v not in (None, "", [], {}, "null", "none") for v in s["tool_args"].values())
    )
    return complete / len(tool_steps)


def f7_efficiency(traj: dict) -> float:
    n_calls = traj["n_tool_calls"]
    if n_calls == 0:
        return 0.0
    expected = max(1, len(traj["query"].split()) // 5)
    return max(0.0, 1.0 - abs(n_calls / expected - 1.0) * 0.3)


# Decoys — surface-level properties with no expected causal link to quality.
# If IRL assigns them substantial weight, θ is unreliable.

def f8_response_verbosity(traj: dict) -> float:
    """Query character length, set before agent acts — should not predict quality."""
    return min(1.0, len(traj.get("query", "")) / 500)


def f9_unique_tool_count(traj: dict) -> float:
    return min(1.0, len(set(traj.get("tool_names_used", []))) / 5)


def f10_trajectory_length_raw(traj: dict) -> float:
    return min(1.0, len(traj.get("steps", [])) / 20)


FEATURE_FNS = [
    f0_tool_selection_accuracy,
    f1_arg_correctness,
    f2_tool_diversity,
    f3_redundancy_avoidance,
    f4_constraint_adherence,
    f5_call_success_rate,
    f6_arg_completeness,
    f7_efficiency,
    f8_response_verbosity,
    f9_unique_tool_count,
    f10_trajectory_length_raw,
]


def compute_features(traj: dict) -> list[float]:
    if not _SCHEMA_LOADED:
        load_tool_schemas()
    return [fn(traj) for fn in FEATURE_FNS]


def process_split(parsed: list[dict], label: str) -> list[dict]:
    results = []
    for traj in parsed:
        results.append({
            "id": traj["id"],
            "domain": traj.get("domain", ""),
            "pass_rate": traj["pass_rate"],
            "features": compute_features(traj),
            "n_tool_calls": traj["n_tool_calls"],
        })
    feat_matrix = np.array([r["features"] for r in results])
    stats_rows = []
    for i, name in enumerate(FEATURE_NAMES):
        col = feat_matrix[:, i]
        stats_rows.append({"feature": name, "mean": round(float(col.mean()), 4),
                           "std": round(float(col.std()), 4),
                           "min": round(float(col.min()), 4), "max": round(float(col.max()), 4)})
    print(f"{label} ({len(results)} trajectories): feature stats saved")
    return results, stats_rows


def validate_tooleval_correlation(results: list[dict], label: str) -> list[dict]:
    pass_rates = np.array([r["pass_rate"] for r in results])
    feat_matrix = np.array([r["features"] for r in results])
    corr_rows = []
    for i, name in enumerate(FEATURE_NAMES):
        col = feat_matrix[:, i]
        tag = "decoy" if i in DECOY_FEATURE_IDX else "quality"
        if col.std() < 1e-6:
            corr_rows.append({"feature": name, "type": tag, "rho": None, "pval": None, "note": "constant"})
            continue
        rho, pval = stats.spearmanr(col, pass_rates)
        corr_rows.append({"feature": name, "type": tag, "rho": round(float(rho), 4), "pval": round(float(pval), 4)})

    qual_matrix = feat_matrix[:, QUALITY_FEATURE_IDX]
    nonconstant = [i for i, col in enumerate(qual_matrix.T) if col.std() > 1e-6]
    cond = None
    if len(nonconstant) >= 2:
        sub = qual_matrix[:, nonconstant]
        eigvals = np.linalg.eigvalsh(np.cov(sub.T))
        eigvals = eigvals[eigvals > 0]
        if len(eigvals) > 1:
            cond = float(eigvals.max() / eigvals.min())

    return corr_rows, cond


def main():
    load_tool_schemas()
    feature_report = {}

    for split in ("expert", "held_out", "suboptimal"):
        in_path = DATA_DIR / f"{split}_parsed.json"
        out_path = DATA_DIR / f"{split}_features.json"
        if not in_path.exists():
            if split == "suboptimal":
                continue
            raise FileNotFoundError(in_path)

        with open(in_path) as f:
            parsed = json.load(f)

        results, stats_rows = process_split(parsed, split)
        corr_rows, cond_number = validate_tooleval_correlation(results, split)

        feature_report[split] = {
            "n": len(results),
            "feature_stats": stats_rows,
            "tooleval_correlation": corr_rows,
            "collinearity_condition_number": cond_number,
        }
        if cond_number and cond_number > 30:
            print(f"WARNING ({split}): high collinearity (condition number={cond_number:.1f}), θ may be poorly identified")

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    with open(OUTPUT_DIR / "feature_report.json", "w") as f:
        json.dump(feature_report, f, indent=2)
    print("feature extraction done. output/feature_report.json")


if __name__ == "__main__":
    main()
