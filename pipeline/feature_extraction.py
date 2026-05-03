"""
About:
    Computes an 8-dimensional feature vector φ(τ) for each trajectory.
    All features are normalised to [0, 1].
"""

import json
import re
from pathlib import Path
import numpy as np
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (DATA_DIR, OUTPUT_DIR, MODELS_DIR, FEATURE_NAMES, N_FEATURES, QUALITY_FEATURE_IDX, DECOY_FEATURE_IDX)
OUTPUT_DIR.mkdir(exist_ok=True)

_TOOL_SCHEMA: dict[str, dict] = {}
_SCHEMA_LOADED = False

def load_tool_schemas(toolenv_path: Path | None = None) -> dict:
    """
    Walk {category}/{tool}.json and build a lookup dict
    """
    global _TOOL_SCHEMA, _SCHEMA_LOADED

    if toolenv_path is None:
        toolenv_path = DATA_DIR / "toolenv" / "tools"

    if not toolenv_path.exists():
        print(f"[feature_extraction] toolenv not found at {toolenv_path}; using heuristic fallbacks")
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
            req = [p.get("name", "").lower() for p in api.get("required_parameters", []) if isinstance(p, dict)]
            opt = [p.get("name", "").lower() for p in api.get("optional_parameters", []) if isinstance(p, dict)]
            method = (api.get("method") or "GET").upper()
            api_index[api_name] = {
                "required_params": req,
                "optional_params": opt,
                "method": method,
                "description": api.get("description", ""),
            }

        entry = {"tool_name": tool_name, "description": description, "api_list": api_index}

        if std_name:
            _TOOL_SCHEMA[std_name] = entry
        if tool_name and tool_name != std_name:
            _TOOL_SCHEMA[tool_name] = entry
        n_loaded += 1

    _SCHEMA_LOADED = True
    print(f"[feature_extraction] Loaded {n_loaded} tool schemas ({len(_TOOL_SCHEMA)} index entries)")
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


def _is_read_method(method: str) -> bool:
    return method.upper() in {"GET", "HEAD", "OPTIONS"}


def _is_write_method(method: str) -> bool:
    return method.upper() in {"POST", "PUT", "PATCH", "DELETE"}


INFO_TOOL_KEYWORDS = {
    "search", "get", "fetch", "retrieve", "query", "lookup", "find",
    "list", "info", "detail", "check", "read", "load",
}
ACTION_TOOL_KEYWORDS = {
    "post", "create", "update", "delete", "send", "submit", "write",
    "book", "buy", "pay", "set", "modify", "cancel",
}
INVASIVE_ARG_PATTERNS = [
    r"password", r"ssn", r"social.?security", r"credit.?card",
    r"card.?number", r"cvv", r"passport", r"license.?number",
    r"birth.?date", r"dob", r"mother", r"secret", r"private.?key",
]
INVASIVE_RE = [re.compile(p, re.I) for p in INVASIVE_ARG_PATTERNS]
CONSTRAINT_KEYWORDS = [
    "free", "cheap", "budget", "no", "without", "exclude", "avoid",
    "only", "must", "require", "limit", "max", "min",
]


def f0_tool_selection_accuracy(traj: dict) -> float:
    """Did the agent call the correct tool(s)?"""
    intended = {
        (item.get("tool_name") or "").lower()
        for item in traj.get("api_list", [])
        if isinstance(item, dict)
    }
    called = [t.lower() for t in traj["tool_names_used"]
              if t and t.lower() not in ("finish", "finishaction")]

    if not called:
        return 0.0

    if intended:
        matches = sum(1 for c in called if any(i in c or c in i for i in intended))
        schema_score = matches / len(called)

        query_words = set(re.findall(r"\b\w{4,}\b", traj["query"].lower()))
        desc_hits = 0
        for tool in called:
            entry = _get_tool_entry(tool)
            if entry:
                desc_words = set(re.findall(r"\b\w{4,}\b", entry["description"].lower()))
                if query_words & desc_words:
                    desc_hits += 1
        desc_bonus = 0.15 * (desc_hits / len(called))
        return min(1.0, schema_score * 0.85 + desc_bonus)
    else:
        query_words = set(re.findall(r"\b\w{4,}\b", traj["query"].lower()))
        hits = sum(1 for t in called if query_words & set(re.split(r"[_\-\s]", t)))
        solved_bonus = 0.2 if traj["finish_type"] == "give_answer" else 0.0
        return min(1.0, (hits / len(called)) * 0.8 + solved_bonus)


def f1_arg_correctness(traj: dict) -> float:
    """Did the agent pass the required parameters for each call?"""
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
            args_lower = {k.lower() for k in args.keys()}
            if required:
                covered = sum(1 for r in required if r in args_lower)
                req_score = covered / len(required)
            else:
                req_score = 1.0 if args else 0.5

            optional = api_entry["optional_params"]
            if optional:
                opt_covered = sum(1 for o in optional if o in args_lower)
                opt_bonus = 0.1 * (opt_covered / len(optional))
            else:
                opt_bonus = 0.0

            scores.append(min(1.0, req_score + opt_bonus))
        else:
            has_value = any(v not in (None, "", [], {}) for v in args.values())
            scores.append(1.0 if (args and has_value) else 0.0)

    return float(np.mean(scores)) if scores else 0.0


def f2_call_ordering(traj: dict) -> float:
    """
    Tool diversity: fraction of tool calls using distinct tool names,
    with penalty for consecutive same-tool repetition (loop failure pattern).

    Replaces schema-dependent GET/POST ordering check. Without toolenv,
    keyword heuristics label almost everything neutral → no violations →
    constant 1.0, zero variance, high collinearity.

    Diversity is computable without schema and discriminates loop failures
    (failed trajectories repeatedly call the same tool) from successful
    ones that use varied tools purposefully.
    """
    PSEUDO = {"finish", "finishaction"}
    tool_steps = [s for s in traj["steps"]
                  if s["role"] == "assistant" and s["tool_name"]
                  and (s["tool_name"] or "").lower() not in PSEUDO]
    if not tool_steps:
        return 0.5

    names = [s["tool_name"].lower() for s in tool_steps]
    diversity = len(set(names)) / len(names)

    # Penalise consecutive same-tool runs > 3
    max_run = cur_run = 1
    for i in range(1, len(names)):
        cur_run = cur_run + 1 if names[i] == names[i-1] else 1
        max_run = max(max_run, cur_run)
    loop_penalty = max(0.0, (max_run - 3) * 0.1)

    return max(0.0, diversity - loop_penalty)


def f3_redundancy_avoidance(traj: dict) -> float:
    """
    Penalise consecutive same-tool repetition.

    DFSDT re-sends the full conversation each turn, so exact (tool, args)
    deduplication over-penalises all trajectories equally. Consecutive
    repetition is the failure signal: a stuck agent calls the same tool
    repeatedly without making progress.
    """
    PSEUDO = {"finish", "finishaction"}
    tool_steps = [s for s in traj["steps"]
                  if s["role"] == "assistant" and s["tool_name"]
                  and (s["tool_name"] or "").lower() not in PSEUDO]
    if not tool_steps:
        return 1.0

    names = [s["tool_name"].lower() for s in tool_steps]
    consecutive = sum(1 for i in range(1, len(names)) if names[i] == names[i-1])
    repeat_rate = consecutive / max(len(names) - 1, 1)
    return 1.0 - repeat_rate


def f4_constraint_adherence(traj: dict) -> float:
    """Check if constraint keywords from the query appear in tool args."""
    query_lower = traj["query"].lower()
    active_constraints = [kw for kw in CONSTRAINT_KEYWORDS if kw in query_lower]

    if not active_constraints:
        return 1.0

    arg_blob = ""
    for s in traj["steps"]:
        args = s.get("tool_args") or {}
        if isinstance(args, dict):
            arg_blob += " ".join(str(v) for v in args.values()).lower()

    hits = sum(1 for kw in active_constraints if kw in arg_blob)
    return hits / len(active_constraints)


def f5_info_sufficiency(traj: dict) -> float:
    """
    Call success rate: fraction of tool calls that received a non-error response.

    Replaces info_sufficiency, which had inverted correlation (ρ=-0.22):
    failed DFSDT trajectories make many repeated calls → high info keyword
    count → high score despite failing. Root cause: keyword matching on tool
    names mistakes iteration for information gathering.

    Call success rate correctly discriminates: successful trajectories receive
    real data; failed ones get empty responses or API errors.
    """
    result_steps = [s for s in traj["steps"]
                    if s["role"] in ("tool", "function", "observation")
                    and s.get("tool_output") is not None]
    if not result_steps:
        return 0.0

    def is_success(output):
        if not output or not output.strip():
            return False
        low = output.lower().strip()
        if low in ("{}", "[]", '""', "null", "none", ""):
            return False
        error_kw = ['"error":', "traceback", "404", "403", "500",
                    "rate limit", "invalid api", "unauthorized"]
        if any(k in low for k in error_kw):
            # Still ok if response field has content
            if '"response"' in low:
                m = re.search('"response":\\s*"([^"]{10,})"', output)
                return bool(m)
            return False
        return len(output.strip()) > 5

    successes = sum(1 for s in result_steps if is_success(s.get("tool_output", "") or ""))
    return successes / len(result_steps)


def f6_invasiveness_minimisation(traj: dict) -> float:
    """
    Argument completeness: fraction of tool calls with at least one
    non-trivial argument value.

    Replaces invasiveness_minimisation (mean≈0.995, near-constant, zero
    signal). ToolBench queries are benign API tasks; sensitive arg patterns
    essentially never appear, giving zero variance and inflating collinearity.

    Argument completeness discriminates: failed trajectories often retry
    tools with empty args {}; successful ones pass meaningful parameter values.
    """
    PSEUDO = {"finish", "finishaction"}
    tool_steps = [s for s in traj["steps"]
                  if s["role"] == "assistant" and s["tool_name"]
                  and (s["tool_name"] or "").lower() not in PSEUDO]
    if not tool_steps:
        return 0.0

    complete = sum(
        1 for s in tool_steps
        if isinstance(s.get("tool_args"), dict)
        and any(v not in (None, "", [], {}, "null", "none")
                for v in s["tool_args"].values())
    )
    return complete / len(tool_steps)


def f7_efficiency(traj: dict) -> float:
    """Penalise excessive calls relative to query complexity."""
    n_calls = traj["n_tool_calls"]
    query_len = len(traj["query"].split())
    expected_calls = max(1, query_len // 5)

    if n_calls == 0:
        return 0.0

    ratio = n_calls / expected_calls
    return max(0.0, 1.0 - abs(ratio - 1.0) * 0.3)

#decoy surface-level trajectory properties with no expected causal relationship to quality. 
def f8_response_verbosity(traj: dict) -> float:
    """
    Decoy: normalised query character length.

    Replaces final_answer length, which correlated with success because
    failed trajectories have empty final_answer while successful ones have
    real text — making it a proxy for outcome, not a neutral decoy.
    Query length is set before the agent acts and varies independently
    of outcome. Expected θ ≈ 0.
    """
    query = traj.get("query") or ""
    return min(1.0, len(query) / 500)



def f9_unique_tool_count(traj: dict) -> float:
    """
    Decoy: number of distinct tool types called, normalised.
    """
    n_unique = len(set(traj.get("tool_names_used", [])))
    # Normalise: cap at 5 distinct tools
    return min(1.0, n_unique / 5)


def f10_trajectory_length_raw(traj: dict) -> float:
    """
    Decoy: raw total step count, normalised.
    Correlates loosely with task complexity but not with quality.
    """
    n_steps = len(traj.get("steps", []))
    return min(1.0, n_steps / 20)


FEATURE_FNS = [
    f0_tool_selection_accuracy,
    f1_arg_correctness,
    f2_call_ordering,
    f3_redundancy_avoidance,
    f4_constraint_adherence,
    f5_info_sufficiency,
    f6_invasiveness_minimisation,
    f7_efficiency,
    f8_response_verbosity,#decoy
    f9_unique_tool_count,#decoy
    f10_trajectory_length_raw,#decoy
]


def compute_features(traj: dict) -> list[float]:
    if not _SCHEMA_LOADED:
        load_tool_schemas()
    return [fn(traj) for fn in FEATURE_FNS]


def process_split(parsed: list[dict], label: str) -> list[dict]:
    results = []
    for traj in parsed:
        feats = compute_features(traj)
        results.append({
            "id": traj["id"],
            "domain": traj.get("domain", ""),
            "pass_rate": traj["pass_rate"],
            "features": feats,
            "n_tool_calls": traj["n_tool_calls"],
        })

    print(f"\n{label} — {len(results)} trajectories")
    feat_matrix = np.array([r["features"] for r in results])
    for i, name in enumerate(FEATURE_NAMES):
        col = feat_matrix[:, i]
        print(f"{name:<30}\nmean={col.mean():.3f}\tstd={col.std():.3f}\tmin={col.min():.3f}\tmax={col.max():.3f}")

    return results


def validate_tooleval_correlation(results: list[dict], label: str):
    print(f"\n{label} correlation with pass_rate:")
    pass_rates = np.array([r["pass_rate"] for r in results])
    feat_matrix = np.array([r["features"] for r in results])

    for i, name in enumerate(FEATURE_NAMES):
        col = feat_matrix[:, i]
        tag = "[DECOY]" if i in DECOY_FEATURE_IDX else ""
        if col.std() < 1e-6:
            print(f"{name:<30}{tag} : (constant, no correlation)")
            continue
        rho, pval = stats.spearmanr(col, pass_rates)
        sig = "**" if pval < 0.05 else ("*" if pval < 0.1 else "")
        print(f"{name:<30}{tag} : rho={rho:+.3f} and p={pval:.3f} {sig}")

    # Collinearity check on quality features only
    # High condition number (>30) means θ will be poorly identified
    qual_matrix = feat_matrix[:, QUALITY_FEATURE_IDX]
    nonconstant = [i for i, col in enumerate(qual_matrix.T) if col.std() > 1e-6]
    if len(nonconstant) >= 2:
        sub = qual_matrix[:, nonconstant]
        cov = np.cov(sub.T)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = eigvals[eigvals > 0]
        cond = float(eigvals.max() / eigvals.min()) if len(eigvals) > 1 else float("inf")
        print(f"\nFeature collinearity (quality features only):")
        print(f"Condition number of covariance matrix: {cond:.1f}")
        if cond > 30:
            print("WARNING: high collinearity. θ weights may be unreliable.")


def main():
    load_tool_schemas()

    for split in ("expert", "held_out", "suboptimal"):
        in_path = DATA_DIR / f"{split}_parsed.json"
        out_path = DATA_DIR / f"{split}_features.json"

        if not in_path.exists():
            if split == "suboptimal":
                print(f"\nsuboptimal_parsed.json not found... skipping")
                continue
            else:
                raise FileNotFoundError(f"{in_path} not found")
        with open(in_path) as f:
            parsed = json.load(f)

        results = process_split(parsed, split)
        validate_tooleval_correlation(results, split)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved in {out_path}")
    print("\nfeature extraction done.")


if __name__ == "__main__":
    main()