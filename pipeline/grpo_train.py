"""
GRPO training for all three reward-conditioned agents.
All conditions use identical hyperparameters for a fair comparison.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from importlib.util import spec_from_file_location, module_from_spec
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, OUTPUT_DIR, MODELS_DIR, BASE_MODEL,
    LORA_R, LORA_ALPHA, LORA_TARGET_MODULES,
    GRPO_ARGS, TOOLRL_WEIGHTS, IRL_REWARD_CLIP,
)

SYSTEM_PROMPT = """You are a tool-using AI assistant. Given a query, you must:
1. Decide which tool to call and what arguments to use.
2. Use the tool result to answer the query.
3. Provide a final answer.

Respond in this format:
[TOOL_CALL] tool_name({"arg1": "value1", "arg2": "value2"})
[TOOL_RESULT] (tool output will appear here)
[ANSWER] your final answer"""

_BAD_TOOL_NAMES = ["placeholder", "tool_name", "example"]


def format_prompt(traj: dict) -> str:
    return f"{SYSTEM_PROMPT}\n\n[QUERY] {traj['query']}"


def _load_feature_module():
    spec = spec_from_file_location("feature_extraction", Path(__file__).parent / "feature_extraction.py")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_completion_to_pseudo_traj(prompt: str, completion: str) -> dict:
    query_match = re.search(r"\[QUERY\]\s*(.*?)(?:\n|$)", prompt)
    query = query_match.group(1).strip() if query_match else ""
    steps = []
    step_idx = 0

    for m in re.finditer(r"\[TOOL_CALL\]\s+(\w+)\((.+?)\)(?=\s*\[|\s*$)", completion, re.DOTALL):
        name = m.group(1).strip()
        if any(bad in name.lower() for bad in _BAD_TOOL_NAMES):
            continue
        try:
            args = json.loads(m.group(2).strip())
            if not isinstance(args, dict):
                args = {}
        except Exception:
            args = {kv.group(1): kv.group(2)
                    for kv in re.finditer(r'"(\w+)"\s*:\s*"([^"]*)"', m.group(2))}
        steps.append({"step_idx": step_idx, "role": "assistant",
                      "tool_name": name, "tool_args": args, "tool_output": None})
        step_idx += 1

    for m in re.finditer(r"\[TOOL_RESULT\]\s*(.*?)(?=\[TOOL_CALL\]|\[ANSWER\]|$)", completion, re.DOTALL):
        output = m.group(1).strip()
        if output:
            steps.append({"step_idx": step_idx, "role": "tool",
                          "tool_name": None, "tool_args": None, "tool_output": output[:300]})
            step_idx += 1
    answer_match = re.search(r"\[ANSWER\]\s*(.*?)$", completion, re.DOTALL)
    final_answer = answer_match.group(1).strip() if answer_match else None
    tool_names = [s["tool_name"] for s in steps if s["role"] == "assistant" and s["tool_name"]]

    return {
        "id": "generated", "query": query, "domain": "", "api": "",
        "pass_rate": 0.0, "steps": steps, "final_answer": final_answer,
        "finish_type": "give_answer" if final_answer else "give_up_and_restart",
        "n_tool_calls": len(tool_names),
        "tool_names_used": list(dict.fromkeys(tool_names)),
        "api_list": [],
    }

def make_binary_reward_fn():
    def reward_fn(completions, **kwargs):
        return [1.0 if "[ANSWER]" in (c if isinstance(c, str) else c[0].get("content", ""))
                else 0.0 for c in completions]
    return reward_fn

def make_toolrl_reward_fn():
    w = TOOLRL_WEIGHTS

    def reward_fn(completions, **kwargs):
        rewards = []
        for completion in completions:
            text = completion if isinstance(completion, str) else completion[0].get("content", "")
            tool_calls = re.findall(r"\[TOOL_CALL\]\s+(\w+)\((.*?)\)", text, re.DOTALL)
            has_answer = "[ANSWER]" in text
            if not tool_calls:
                rewards.append(w["outcome"] * (1.0 if has_answer else 0.0))
                continue
            name_scores, schema_scores, value_scores = [], [], []
            for name, args_str in tool_calls:
                name_scores.append(0.0 if any(b in name.lower() for b in _BAD_TOOL_NAMES) else 1.0)
                try:
                    args = json.loads(args_str)
                    valid = isinstance(args, dict) and len(args) > 0
                    schema_scores.append(1.0 if valid else 0.0)
                    non_empty = sum(1 for v in args.values() if v not in (None, "", [], {})) if valid else 0
                    value_scores.append(non_empty / max(len(args), 1) if valid else 0.0)
                except Exception:
                    schema_scores.append(0.0)
                    value_scores.append(0.0)
            rewards.append(
                w["outcome"] * (1.0 if has_answer else 0.0)
                + w["name"] * float(np.mean(name_scores))
                + w["schema"] * float(np.mean(schema_scores))
                + w["values"] * float(np.mean(value_scores))
            )
        return rewards
    return reward_fn

def make_irl_reward_fn(theta: np.ndarray):
    r_min, r_max = IRL_REWARD_CLIP
    compute_features = _load_feature_module().compute_features

    def reward_fn(completions, **kwargs):
        prompts = kwargs.get("prompts", [""] * len(completions))
        rewards = []
        for prompt, completion in zip(prompts, completions):
            text = completion if isinstance(completion, str) else completion[0].get("content", "")
            traj = parse_completion_to_pseudo_traj(prompt, text)
            raw = float(theta @ np.array(compute_features(traj)))
            rewards.append((np.clip(raw, r_min, r_max) - r_min) / (r_max - r_min))
        return rewards
    return reward_fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["binary", "toolrl", "irl"], default="binary")
    args = parser.parse_args()
    output_dir = MODELS_DIR / args.condition
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(DATA_DIR / "expert_parsed.json") as f:
        expert = json.load(f)
    prompts = [format_prompt(t) for t in expert]
    print(f"condition={args.condition} training_examples={len(prompts)}")

    has_gpu = torch.cuda.is_available()
    if has_gpu:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb < 6:
            print(f"insufficient VRAM ({vram_gb:.1f}GB)")
            has_gpu = False

    if not has_gpu:
        print("no GPU. saving config")
        with open(output_dir / "training_config.json", "w") as f:
            json.dump({"condition": args.condition, "base_model": BASE_MODEL,
                       "grpo_args": GRPO_ARGS, "n_train_examples": len(prompts)}, f, indent=2)
        with open(DATA_DIR / f"grpo_dataset_{args.condition}.json", "w") as f:
            json.dump([{"prompt": p, "traj_id": t["id"]} for p, t in zip(prompts, expert)], f)
        return

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map="auto")
    if args.condition == "binary":
        reward_fn = make_binary_reward_fn()
    elif args.condition == "toolrl":
        reward_fn = make_toolrl_reward_fn()
    else:
        with open(OUTPUT_DIR / "theta_weights.json") as f:
            theta = np.array(json.load(f)["theta"])
        reward_fn = make_irl_reward_fn(theta)

    grpo_config = GRPOConfig(
        **{k: v for k, v in GRPO_ARGS.items() if k != "max_new_tokens"},
        output_dir=str(output_dir),
        max_completion_length=GRPO_ARGS.get("max_new_tokens", 512),
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=Dataset.from_list([{"prompt": p} for p in prompts]),
        peft_config=LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
                               target_modules=LORA_TARGET_MODULES, task_type="CAUSAL_LM"),
    )

    print(f"starting GRPO training ({args.condition})...")
    trainer.train()
    trainer.save_model(str(output_dir))

    log = [{"step": e["step"], "loss": e.get("loss"), "reward": e.get("reward")}
           for e in trainer.state.log_history]
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"model saved to {output_dir}")

if __name__ == "__main__":
    main()
