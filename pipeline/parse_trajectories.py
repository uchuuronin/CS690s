"""
Parses raw ToolBench trajectories into a clean step-level format.
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
    try:
        obj = json.loads(content)
        name = obj.get("name") or obj.get("tool") or obj.get("function")
        args = obj.get("arguments") or obj.get("parameters") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"raw": args}
        return name, args
    except Exception:
        pass

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
    steps = []
    for step_idx, turn in enumerate(train_messages):
        if not isinstance(turn, list):
            turn = [turn]
        for msg in turn:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            content = msg.get("content") or ""

            step = {
                "step_idx": step_idx,
                "role": role,
                "tool_name": None,
                "tool_args": None,
                "tool_output": None,
                "raw_content": content[:2000],
            }

            if role == "assistant":
                fc = msg.get("function_call")
                if fc and isinstance(fc, dict):
                    step["tool_name"] = fc.get("name")
                    args = fc.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {"raw": args}
                    step["tool_args"] = args
                else:
                    step["tool_name"], step["tool_args"] = parse_tool_call(content)

            elif role in ("tool", "function", "observation"):
                step["tool_output"] = content[:500]
                step["tool_name"] = msg.get("name") or msg.get("tool_call_id")
            steps.append(step)
    return steps


def parse_trajectory(raw: dict) -> dict:
    answer = raw.get("answer_generation", {}) or raw.get("answer", {})
    if not isinstance(answer, dict):
        answer = {}
    steps = parse_messages(answer.get("train_messages", []))
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

    parsed, skipped = [], 0
    for raw in raw_list:
        try:
            p = parse_trajectory(raw)
            if p["n_tool_calls"] == 0:
                skipped += 1
                continue
            parsed.append(p)
        except Exception as e:
            skipped += 1
            print(f"skipping {raw.get('id')}: {e}")

    with open(out_path, "w") as f:
        json.dump(parsed, f, indent=2)
    avg_calls = sum(t["n_tool_calls"] for t in parsed) / max(len(parsed), 1)
    print(f"{label}: {len(parsed)} parsed ({skipped} skipped), avg_calls={avg_calls:.1f}")
    return parsed


def main():
    process_file(DATA_DIR / "expert_trajectories.json", DATA_DIR / "expert_parsed.json", "expert")
    process_file(DATA_DIR / "held_out_trajectories.json", DATA_DIR / "held_out_parsed.json", "held_out")
    if (DATA_DIR / "suboptimal_trajectories.json").exists():
        process_file(DATA_DIR / "suboptimal_trajectories.json", DATA_DIR / "suboptimal_parsed.json", "suboptimal")

if __name__ == "__main__":
    main()