"""
Source Copyright:
    ToolBench is distributed via Google Drive (official):
    https://drive.google.com/drive/folders/1TysbSWYpP8EioFu9xPJtpbJZMLLmwAmL

About:
    Downloads tool bench and filter into expert/held-out/suboptimal splits.

    Used the per-query answer files from data/answer/G{1,2,3}_answer/ which contain the raw DFSDT trajectories with solvability labels.
    Solvability: "finish_type" == "give_answer" is the reliable proxy for pass_rate, since the ToolEval pass_rate script requires OpenAI calls and isn't stored per-file.
    Used G1 (single-tool with one API per query) for the project.
"""

import json
import os
import random
from pathlib import Path
from collections import defaultdict
import argparse

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (DATA_DIR, OUTPUT_DIR, PASS_RATE_THRESHOLD, TARGET_EXPERT, TARGET_HELD_OUT)

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULT_GROUP = "G1"

def compute_pass_rate(answer_data: dict) -> float:
    """
    Using finish_type as a binary proxy where 
        1= give_answer (agent produced a final answer) and 
        0= give_up_and_restart (agent gave up)
    """
    answer =answer_data.get("answer", {})
    if not isinstance(answer, dict):
        return 0.0
    finish =answer.get("finish_type", "")
    if finish == "give_answer":
        return 1.0
    elif finish == "give_up_and_restart":
        return 0.0
    else:
        return 0.5 # partial/unknown


def is_valid_trajectory(answer_data: dict) -> bool:
    """Must have at least one tool call in the trajectory."""
    answer = answer_data.get("answer", {})
    if not isinstance(answer,dict):
        return False
    train_messages = answer.get("train_messages", [])
    for turn in train_messages:
        if not isinstance(turn,list):
            turn = [turn]
        for msg in turn:
            if isinstance(msg,dict) and msg.get("role") == "tool":
                return True
    return False


def load_from_local(data_root: Path, group: str, categories: list[str], target_expert: int, target_held_out: int, target_suboptimal: int, threshold: float):
    answer_dir = data_root / "answer" / f"{group}_answer"
    if not answer_dir.exists():
        raise FileNotFoundError(f"Answer directory not found: {answer_dir}\nDownload ToolBench data locally first")

    #load instruction file for query text
    instr_file = data_root / "instruction" / f"{group}_query.json"
    query_map = {}
    if instr_file.exists():
        with open(instr_file) as f:
            instrs = json.load(f)
        for item in instrs:
            query_map[str(item.get("query_id", ""))] = item.get("query", "")

    expert,held_out,suboptimal = [], [], []
    total_seen = total_valid = 0
    by_category = defaultdict(int)

    print(f"Going through answer_dir={answer_dir} ...")
    json_files = list(answer_dir.rglob("*_DFS_woFilter_w2.json"))
    random.shuffle(json_files)#for random sampling

    for fpath in json_files:
        total_seen += 1

        # Check if we have enough
        exp_done = len(expert) >= target_expert * 3
        held_done = len(held_out) >= target_held_out * 3
        sub_done = len(suboptimal) >= target_suboptimal
        if exp_done and held_done and sub_done:
            break

        try:
            with open(fpath) as f:
                raw = json.load(f)
        except Exception:
            continue

        if not is_valid_trajectory(raw):
            continue

        total_valid += 1

        # infer category from path or api_list
        category = None
        api_list = raw.get("api_list", []) or raw.get("available_tools", [])
        if api_list and isinstance(api_list[0], dict):
            category = api_list[0].get("category_name", "")
        if not category:
            parts = fpath.parts
            for part in parts:
                if part in categories:
                    category = part
                    break

        query_id = fpath.stem.split("_")[0]
        query = raw.get("query", "") or query_map.get(query_id, "")

        pr = compute_pass_rate(raw)
        traj = {"id": query_id,
                "query": query,
                "category": category or "unknown",
                "group": group,
                "pass_rate": pr,
                "answer": raw.get("answer", {}),
                "api_list": api_list
                }
        by_category[category] += 1

        if pr>= threshold:
            if not exp_done:
                expert.append(traj)
            if not held_done:
                held_out.append(traj)
        else:
            if not sub_done:
                suboptimal.append(traj)
            # Also add some suboptimal to held-out for quality variety
            if not held_done and random.random() < 0.3:
                held_out.append(traj)

        if total_seen % 1000 == 0:
            print(f"Seen={total_seen}; Valid={total_valid};\nExpert count={len(expert)}, Held-out count={len(held_out)}, Suboptimal count={len(suboptimal)};\nBy category={dict(by_category)}")

    return expert, held_out, suboptimal, total_seen, total_valid, dict(by_category)

def main(data_root, group, target_expert, target_held_out, threshold):
    target_suboptimal = target_expert * 2
    categories = [] # all G1 categories used

    print(f"Group: {group}, Expert target: {target_expert}, Held-out target: {target_held_out}, Suboptimal target: {target_suboptimal}")
    print(f"Pass rate threshold: {threshold} (finish_type= 1 if give_answer, 0 if give_up)")

    if data_root is None:
        data_root = DATA_DIR
    else:
        data_root = Path(data_root)
    expert, held_out, suboptimal, total_seen, total_valid, by_category = load_from_local(data_root, group, categories, target_expert, target_held_out, target_suboptimal, threshold)
 
    #Final shuffle and trim
    random.shuffle(expert)
    random.shuffle(held_out)
    random.shuffle(suboptimal)
    expert = expert[:target_expert]
    held_out = held_out[:target_held_out]
    suboptimal = suboptimal[:target_suboptimal]
 
    if len(expert) < 50:
        print(f"\nWARNING: Only {len(expert)} expert trajectories found, either lower --pass-rate or switch --group")
 
    with open(DATA_DIR / "expert_trajectories.json", "w") as f:
        json.dump(expert, f, indent=2)
    with open(DATA_DIR / "held_out_trajectories.json", "w") as f:
        json.dump(held_out, f, indent=2)
    with open(DATA_DIR / "suboptimal_trajectories.json", "w") as f:
        json.dump(suboptimal, f, indent=2)
        
    print(f"\nGenerated:")
    print(f"data/expert_trajectories.json ({len(expert)})")
    print(f"data/held_out_trajectories.json ({len(held_out)}, mixed quality)")
    print(f"data/suboptimal_trajectories.json ({len(suboptimal)}, for IRL model dist.)")
    print("\ndata filtering done.")
    
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,description=__doc__)
    parser.add_argument("--data-root", type=str, default=None, help="Path to ToolBench data (Defaults to data/).")
    parser.add_argument("--group", type=str, default=DEFAULT_GROUP, choices=["G1", "G2", "G3"],help="G1=single-tool (default), G2=intra-category multi-tool, G3=intra-collection multi-tool.")
    parser.add_argument("--n-expert", type=int, default=TARGET_EXPERT)
    parser.add_argument("--n-held-out", type=int, default=TARGET_HELD_OUT)
    parser.add_argument("--pass-rate", type=float, default=PASS_RATE_THRESHOLD, help="threshold that controls expert/suboptimal split.")
    cli = parser.parse_args()
 
    main(data_root=cli.data_root,
         group=cli.group,
         target_expert=cli.n_expert,
         target_held_out=cli.n_held_out,
         threshold=cli.pass_rate)