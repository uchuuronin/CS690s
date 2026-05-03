"""
About:
    Parse raw ToolBench trajectories into a clean structured format.
"""

import json
import re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR


def parse_tool_call(content: str) -> tuple[str | None, dict | None]:
    if not content:
        return None, None

    # Try JSON block extraction
    try:
        obj = json.loads(content)
        name = obj.get("name") or obj.get("tool") or obj.get("function")
        args = obj.get("arguments") or obj.get("parameters") or {}
        if isinstance(args, str):
            try:
                args =json.loads(args)
            except Exception:
                args ={"raw": args}
        return name, args
    except Exception:
        pass

    # Try regex parsing
    m = re.search(r'"name"\s*:\s*"([^"]+)"', content)
    if m:
        name = m.group(1)
        arg_m = re.search(r'"arguments"\s*:\s*(\{.*?\})', content, re.DOTALL)
        args = {}
        if arg_m:
            try:
                args = json.loads(arg_m.group(1))
            except Exception:
                args = {"raw": arg_m.group(1)}
        return name, args

    return None, None


def parse_messages(train_messages: list) -> list[dict]:
    # Flatten train_messages and parse into step dicts.
    steps = []
    step_idx = 0

    for turn in train_messages:
        if not isinstance(turn, list):
            turn = [turn]
        for msg in turn:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            content = msg.get("content") or ""

            step = {"step_idx": step_idx,
                    "role": role,
                    "tool_name": None,
                    "tool_args": None,
                    "tool_output": None,
                    "raw_content": content[:2000] # truncate large tool outputs
                    }

            if role == "assistant":
                # Check function_call field first (OpenAI format used by ToolBench)
                fc = msg.get("function_call")
                if fc and isinstance(fc, dict):
                    step["tool_name"] = fc.get("name")
                    args = fc.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args= {"raw": args}
                    step["tool_args"] = args
                else:
                    tool_name, tool_args = parse_tool_call(content)
                    step["tool_name"] = tool_name
                    step["tool_args"] = tool_args

            elif role in ("tool", "function", "observation"):
                # Tool response
                step["tool_output"] = content[:500]
                step["tool_name"] = msg.get("name") or msg.get("tool_call_id")

            elif role == "system" or role == "user":
                pass # keep raw_content only

            steps.append(step)
            step_idx += 1

    return steps


def parse_trajectory(raw: dict) -> dict:
    answer = raw.get("answer_generation", {}) or raw.get("answer", {})
    if not isinstance(answer, dict):
        answer = {}

    train_messages = answer.get("train_messages", [])
    steps = parse_messages(train_messages)

    tool_calls = [s for s in steps if s["role"] == "assistant" and s["tool_name"]]
    tool_names_used = list(dict.fromkeys(s["tool_name"] for s in tool_calls if s["tool_name"]))

    final_answer = None
    for s in reversed(steps):
        if s["role"] == "assistant" and not s["tool_name"]:
            final_answer = s["raw_content"]
            break

    return {
        "id": str(raw.get("id", "")),
        "query": raw.get("query", ""),
        "domain": raw.get("domain", ""),
        "api": raw.get("api", ""),
        "pass_rate": raw.get("pass_rate", 0.0),
        "steps": steps,
        "final_answer": final_answer,
        "finish_type": answer.get("finish_type", "unknown"),
        "n_tool_calls": len(tool_calls),
        "tool_names_used": tool_names_used,
    }

def process_file(in_path: Path, out_path: Path, label: str):
    with open(in_path) as f:
        raw_list = json.load(f)

    parsed = []
    skipped = 0
    for raw in raw_list:
        try:
            p = parse_trajectory(raw)
            if p["n_tool_calls"] == 0:
                skipped += 1
                continue
            parsed.append(p)
        except Exception as e:
            skipped += 1
            print(f"Skipping trajectory {raw.get('id')}: {e}")

    with open(out_path, "w") as f:
        json.dump(parsed, f, indent=2)
    print(f"{label}: {len(parsed)} parsed, {skipped} skipped == {out_path}")

    # Quick summary
    avg_steps = sum(t["n_tool_calls"] for t in parsed) / max(len(parsed), 1)
    domains = {}
    for t in parsed:
        domains[t["domain"]] = domains.get(t["domain"], 0) + 1
    print(f"Avg tool calls per trajectory: {avg_steps:.1f}")
    print(f"\tBy domain: {domains}")

    return parsed


def main():
    expert = process_file(DATA_DIR / "expert_trajectories.json",DATA_DIR / "expert_parsed.json","Expert",)
    held_out = process_file(DATA_DIR / "held_out_trajectories.json",DATA_DIR / "held_out_parsed.json","Held-out",)
    subopt_raw = DATA_DIR / "suboptimal_trajectories.json"
    if subopt_raw.exists():
        process_file(subopt_raw,DATA_DIR / "suboptimal_parsed.json","Suboptimal")
    else:
        print("suboptimal_trajectories.json not found — will be created by Stage 1")
    print("\nparsing trajectories done.")


if __name__ == "__main__":
    main()