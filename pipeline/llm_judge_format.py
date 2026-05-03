"""
Two modes:
  format  -- reads eval_completions_{condition}.json for each condition,
              samples n trajectories per condition, strips condition labels,
              shuffles, and writes a formatted prompt you paste into any LLM chat.
              Also writes a key file so you can match ratings back.

  score   -- reads the ratings you paste back (as a simple JSON or CSV),
              joins them to the key, and writes output/llm_judge_results.json.
"""

import json
import random
import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, OUTPUT_DIR, MODELS_DIR

CONDITIONS = ["sft", "binary", "toolrl", "irl"]
IMPLICIT_DIMENSIONS = [
    ("constraint_adherence", "Did the agent reflect constraints mentioned in the query in its tool arguments? (e.g. if query says 'free only', did args filter for free?)"),
    ("redundancy_avoidance", "Did the agent avoid repeating the same tool call unnecessarily?"),
    ("efficiency", "Did the agent use an appropriate number of tool calls, or did it over-call or under-call?"),
    ("answer_quality", "Did the final answer actually address the query?"),
]

JUDGE_PROMPT_HEADER = """You are evaluating tool-using AI agent trajectories. Each trajectory shows a query the agent received, the tool calls it made, and its final answer.

Rate each trajectory on the four dimensions below using a 1-5 scale:
  1 = very poor
  2 = poor  
  3 = acceptable
  4 = good
  5 = excellent

Dimensions:
"""

JUDGE_PROMPT_FORMAT = """For each trajectory, respond with ONLY a JSON block in this exact format (no extra text):

{
  "trajectory_id": "<id>",
  "constraint_adherence": <1-5>,
  "redundancy_avoidance": <1-5>,
  "efficiency": <1-5>,
  "answer_quality": <1-5>,
  "notes": "<one sentence optional>"
}

Process all trajectories and return a JSON array of these objects.
"""


def format_trajectory_for_judge(query: str, completion: str, traj_id: str) -> str:
    lines = [f"--- TRAJECTORY {traj_id} ---", f"QUERY: {query}", ""]

    # parse tool calls
    import re
    tool_calls = re.findall(r"\[TOOL_CALL\]\s+(\w+)\((.+?)\)(?=\s*\[|\s*$)", completion, re.DOTALL)
    tool_results = re.findall(r"\[TOOL_RESULT\]\s*(.*?)(?=\[TOOL_CALL\]|\[ANSWER\]|$)", completion, re.DOTALL)
    answer_m = re.search(r"\[ANSWER\]\s*(.*?)$", completion, re.DOTALL)

    if not tool_calls:
        lines.append("(no tool calls made)")
    else:
        for i, (name, args_str) in enumerate(tool_calls):
            try:
                import json as _json
                args = _json.loads(args_str.strip())
                args_display = _json.dumps(args, indent=2)
            except Exception:
                args_display = args_str.strip()
            lines.append(f"Tool call {i+1}: {name}")
            lines.append(f"  Args: {args_display}")
            if i < len(tool_results):
                result = tool_results[i].strip()[:200]
                lines.append(f"  Result: {result}")
            lines.append("")

    if answer_m:
        lines.append(f"Final answer: {answer_m.group(1).strip()[:300]}")
    else:
        lines.append("Final answer: (none — agent did not produce an answer)")

    lines.append("")
    return "\n".join(lines)


def format_mode(n_per_condition: int, seed: int = 42):
    random.seed(seed)

    with open(DATA_DIR / "held_out_parsed.json") as f:
        held_out = json.load(f)
    queries = {t["id"]: t["query"] for t in held_out}
    query_list = [t["query"] for t in held_out]

    samples = []  # list of (traj_id, condition, query, completion)

    for condition in CONDITIONS:
        cache_path = OUTPUT_DIR / f"eval_completions_{condition}.json"
        if not cache_path.exists():
            print(f"no completions for {condition}, skipping")
            continue
        with open(cache_path) as f:
            completions = json.load(f)

        n = min(n_per_condition, len(completions))
        indices = random.sample(range(len(completions)), n)
        for idx in indices:
            query = query_list[idx] if idx < len(query_list) else f"query_{idx}"
            samples.append({
                "condition": condition,
                "original_index": idx,
                "query": query,
                "completion": completions[idx],
            })

    random.shuffle(samples)

    # assign blind ids
    key = []
    formatted_blocks = []
    for i, s in enumerate(samples):
        traj_id = f"T{i+1:03d}"
        key.append({
            "traj_id": traj_id,
            "condition": s["condition"],
            "original_index": s["original_index"],
        })
        formatted_blocks.append(
            format_trajectory_for_judge(s["query"], s["completion"], traj_id)
        )

    # save key (never share this until after ratings collected)
    key_path = OUTPUT_DIR / "llm_judge_key.json"
    with open(key_path, "w") as f:
        json.dump(key, f, indent=2)
    print(f"key saved to {key_path} (do not share this until ratings are collected)")

    # write the prompt file to paste into LLM
    prompt_path = OUTPUT_DIR / "llm_judge_prompt.txt"
    with open(prompt_path, "w") as f:
        f.write(JUDGE_PROMPT_HEADER)
        for dim, desc in IMPLICIT_DIMENSIONS:
            f.write(f"  {dim}: {desc}\n")
        f.write("\n")
        f.write(JUDGE_PROMPT_FORMAT)
        f.write("\n\n")
        f.write("=" * 60 + "\n\n")
        f.write("\n".join(formatted_blocks))

    print(f"prompt saved to {prompt_path}")
    print(f"total trajectories: {len(samples)} ({n_per_condition} per condition)")
    print(f"paste the contents of llm_judge_prompt.txt into your LLM chat")
    print(f"save the JSON array it returns to a file, then run:")
    print(f"  python pipeline/llm_judge_format.py score --ratings your_ratings.json")


def score_mode(ratings_path: str):
    with open(ratings_path) as f:
        ratings = json.load(f)

    key_path = OUTPUT_DIR / "llm_judge_key.json"
    if not key_path.exists():
        print(f"key not found at {key_path}. run format mode first")
        return
    with open(key_path) as f:
        key = json.load(f)

    key_by_id = {k["traj_id"]: k for k in key}

    # join ratings to key
    joined = []
    for r in ratings:
        traj_id = r.get("trajectory_id")
        if not traj_id or traj_id not in key_by_id:
            print(f"unmatched traj_id: {traj_id}")
            continue
        entry = {**key_by_id[traj_id], **{k: v for k, v in r.items() if k != "trajectory_id"}}
        joined.append(entry)

    # aggregate by condition
    import numpy as np
    dimensions = ["constraint_adherence", "redundancy_avoidance", "efficiency", "answer_quality"]
    by_condition = {}
    for condition in CONDITIONS:
        cond_entries = [j for j in joined if j["condition"] == condition]
        if not cond_entries:
            continue
        by_condition[condition] = {
            "n": len(cond_entries),
        }
        for dim in dimensions:
            vals = [e[dim] for e in cond_entries if isinstance(e.get(dim), (int, float))]
            if vals:
                by_condition[condition][f"{dim}_mean"] = round(float(np.mean(vals)), 3)
                by_condition[condition][f"{dim}_std"] = round(float(np.std(vals)), 3)

    results = {
        "by_condition": by_condition,
        "all_ratings": joined,
        "dimensions": dimensions,
        "note": "blind evaluation: LLM judge did not know which condition produced each trajectory",
    }

    out_path = OUTPUT_DIR / "llm_judge_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {out_path}")

    # quick summary
    print("\nper-condition means:")
    for condition, stats in by_condition.items():
        means = {d: stats.get(f"{d}_mean", "n/a") for d in dimensions}
        print(f"  {condition}: {means}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode")

    fmt = sub.add_parser("format")
    fmt.add_argument("--n", type=int, default=20, help="trajectories per condition")
    fmt.add_argument("--seed", type=int, default=42)

    sc = sub.add_parser("score")
    sc.add_argument("--ratings", required=True, help="path to JSON file with LLM ratings")

    cli = parser.parse_args()
    if cli.mode == "format":
        format_mode(cli.n, cli.seed)
    elif cli.mode == "score":
        score_mode(cli.ratings)
    else:
        parser.print_help()
