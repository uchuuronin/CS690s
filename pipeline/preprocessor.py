"""
Inspired by: ToolLLM (Qin et al., ICLR 2024) — ToolBench dataset and DFSDT trajectory format.


Filters ToolBench into expert / held-out / suboptimal splits using finish_type
as a binary pass-rate proxy (give_answer=1.0, give_up_and_restart=0.0).
Also extracts DFSDT alternative branches as additional suboptimal trajectories
for the IRL model distribution.
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, OUTPUT_DIR, PASS_RATE_THRESHOLD, TARGET_EXPERT, TARGET_HELD_OUT

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULT_GROUP = "G1"


def compute_pass_rate(answer_data: dict) -> float:
    ag = answer_data.get("answer_generation", {})
    if isinstance(ag, dict):
        finish = ag.get("finish_type", "")
        if finish == "give_answer":
            return 1.0
        elif finish == "give_up_and_restart":
            return 0.0
    if answer_data.get("win") is True:
        return 1.0
    if answer_data.get("win") is False:
        return 0.0
    return 0.5


def is_valid_trajectory(answer_data: dict) -> bool:
    ag = answer_data.get("answer_generation", {})
    if not isinstance(ag, dict):
        return False

    has_tool_call = False
    has_tool_result = False
    for turn in ag.get("train_messages", []):
        if not isinstance(turn, list):
            turn = [turn]
        for msg in turn:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            if role == "assistant":
                if msg.get("function_call"):
                    has_tool_call = True
                else:
                    c = msg.get("content", "") or ""
                    if '"name"' in c and '"arguments"' in c:
                        has_tool_call = True
            elif role in ("tool", "function", "observation"):
                if (msg.get("content", "") or "").strip():
                    has_tool_result = True

    if not has_tool_call:
        return False
    finish = ag.get("finish_type", "")
    if finish == "give_answer":
        if not has_tool_result:
            return False
        if len((ag.get("final_answer", "") or "").strip()) < 10:
            return False
    return True


def extract_dfsdt_branches(raw: dict, query_id: str, query: str, category: str, group: str) -> list[dict]:
    """
    Extract the main DFSDT path plus any stored alternative/failed branches.
    Alternative branches serve as query-matched negatives for the IRL model distribution,
    which is a stronger contrast than cross-query suboptimal trajectories.
    """
    answer = raw.get("answer_generation", {})
    if not isinstance(answer, dict):
        return []

    api_list = raw.get("api_list", []) or raw.get("available_tools", [])
    pr = 1.0 if answer.get("finish_type") == "give_answer" else 0.0

    branches = [{
        "id": query_id,
        "query": query,
        "category": category or "unknown",
        "group": group,
        "pass_rate": pr,
        "answer": answer,
        "api_list": api_list,
        "branch_type": "main",
    }]

    for key in ("failed_paths", "alternative_paths", "explored_nodes"):
        for i, alt in enumerate(raw.get(key, []) or []):
            if not isinstance(alt, dict):
                continue
            alt_answer = alt if "train_messages" in alt else {
                "train_messages": alt.get("messages", []),
                "finish_type": "give_up_and_restart",
            }
            branches.append({
                "id": f"{query_id}_branch_{i}",
                "query": query,
                "category": category or "unknown",
                "group": group,
                "pass_rate": 0.0,
                "answer": alt_answer,
                "api_list": api_list,
                "branch_type": "alternative",
            })

    return branches


def load_from_local(data_root: Path, group: str, target_expert: int,
                    target_held_out: int, target_suboptimal: int, threshold: float):
    answer_dir = data_root / "answer" / f"{group}_answer"
    if not answer_dir.exists():
        raise FileNotFoundError(f"Answer directory not found: {answer_dir}")

    instr_file = data_root / "instruction" / f"{group}_query.json"
    query_map = {}
    if instr_file.exists():
        with open(instr_file) as f:
            for item in json.load(f):
                query_map[str(item.get("query_id", ""))] = item.get("query", "")

    expert, held_out, suboptimal = [], [], []
    by_category = defaultdict(int)
    total_seen = total_valid = 0

    json_files = list(answer_dir.rglob("*_DFS_woFilter_w2.json")) or \
                 list(answer_dir.rglob("*_ChatGPT_DFS_woFilter_w2.json"))
    random.shuffle(json_files)

    for fpath in json_files:
        total_seen += 1
        if len(expert) >= target_expert * 3 and \
           len(held_out) >= target_held_out * 3 and \
           len(suboptimal) >= target_suboptimal:
            break

        try:
            with open(fpath) as f:
                raw = json.load(f)
        except Exception:
            continue

        if not is_valid_trajectory(raw):
            continue
        total_valid += 1

        ag = raw.get("answer_generation", {})
        category = None
        api_list = raw.get("api_list", []) or raw.get("available_tools", [])
        if api_list and isinstance(api_list[0], dict):
            category = api_list[0].get("category_name", "")

        query_id = fpath.stem.split("_")[0]
        query = ag.get("query", "") or raw.get("query", "") or query_map.get(query_id, "")
        if not api_list:
            fa = raw.get("forward_args", {})
            api_list = fa.get("api_list", []) if isinstance(fa, dict) else []

        pr = compute_pass_rate(raw)
        by_category[category] += 1

        branches = extract_dfsdt_branches({**raw, "answer_generation": ag},
                                          query_id, query, category, group)
        main_traj = branches[0]

        if pr >= threshold:
            if len(expert) < target_expert * 3:
                expert.append(main_traj)
            if len(held_out) < target_held_out * 3:
                held_out.append(main_traj)
        else:
            if len(suboptimal) < target_suboptimal:
                suboptimal.append(main_traj)
            if len(held_out) < target_held_out * 3 and random.random() < 0.3:
                held_out.append(main_traj)

        for branch in branches[1:]:
            if len(suboptimal) < target_suboptimal:
                suboptimal.append(branch)

        if total_seen % 1000 == 0:
            print(f"seen={total_seen} valid={total_valid} | expert={len(expert)} held_out={len(held_out)} suboptimal={len(suboptimal)}")

    return expert, held_out, suboptimal, dict(by_category)


def main(data_root, group, target_expert, target_held_out, threshold):
    target_suboptimal = target_expert * 2
    data_root = Path(data_root) if data_root else DATA_DIR

    print(f"group={group} expert={target_expert} held_out={target_held_out} suboptimal={target_suboptimal} threshold={threshold}")

    expert, held_out, suboptimal, by_category = load_from_local(
        data_root, group, target_expert, target_held_out, target_suboptimal, threshold
    )

    random.shuffle(expert); expert = expert[:target_expert]
    random.shuffle(held_out); held_out = held_out[:target_held_out]
    random.shuffle(suboptimal); suboptimal = suboptimal[:target_suboptimal]

    if len(expert) < 50:
        print(f"WARNING: only {len(expert)} expert trajectories — try lowering --pass-rate or switching --group")

    with open(DATA_DIR / "expert_trajectories.json", "w") as f:
        json.dump(expert, f, indent=2)
    with open(DATA_DIR / "held_out_trajectories.json", "w") as f:
        json.dump(held_out, f, indent=2)
    with open(DATA_DIR / "suboptimal_trajectories.json", "w") as f:
        json.dump(suboptimal, f, indent=2)

    print(f"expert={len(expert)} held_out={len(held_out)} suboptimal={len(suboptimal)}")
    print(f"by_category={by_category}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--group", type=str, default=DEFAULT_GROUP, choices=["G1", "G2", "G3"])
    parser.add_argument("--n-expert", type=int, default=TARGET_EXPERT)
    parser.add_argument("--n-held-out", type=int, default=TARGET_HELD_OUT)
    parser.add_argument("--pass-rate", type=float, default=PASS_RATE_THRESHOLD)
    cli = parser.parse_args()
    main(cli.data_root, cli.group, cli.n_expert, cli.n_held_out, cli.pass_rate)