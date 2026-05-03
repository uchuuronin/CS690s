"""
About:
    GRPO training for all three reward conditioned agents.
    Each condition is trained identically (same LR, batch size, steps) to ensure fair comparison.
"""
import argparse
import json
import os
import sys
import re
import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec as _mfs2
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, OUTPUT_DIR, MODELS_DIR, BASE_MODEL, LORA_R, LORA_ALPHA, 
    LORA_DROPOUT, LORA_TARGET_MODULES, GRPO_ARGS, IRL_REWARD_MIN, IRL_REWARD_MAX,
)
MODEL_BASE= MODELS_DIR
SYSTEM_PROMPT= """You are a tool-using AI assistant. Given a query, you must:
1. Decide which tool to call and what arguments to use.
2. Use the tool result to answer the query.
3. Provide a final answer.

Respond in this format:
[TOOL_CALL] tool_name({"arg1": "value1", "arg2": "value2"})
[TOOL_RESULT] (tool output will appear here)
[ANSWER] your final answer"""


def format_prompt(traj: dict) -> str:
    return f"{SYSTEM_PROMPT}\n\n[QUERY] {traj['query']}"


def format_reference(traj: dict) -> str:
    # Reference completion for reward computation.
    parts = []
    for step in traj["steps"]:
        if step["role"] == "assistant" and step["tool_name"]:
            args_str = json.dumps(step["tool_args"] or {})
            parts.append(f"[TOOL_CALL] {step['tool_name']}({args_str})")
        elif step["role"] == "tool":
            out = (step["tool_output"] or "")[:200]
            parts.append(f"[TOOL_RESULT] {out}")
    if traj.get("final_answer"):
        parts.append(f"[ANSWER] {traj['final_answer'][:300]}")
    return "\n".join(parts)

# GRPO reward functions receive (prompts, completions, **kwargs) and return a list of scalar rewards.
def make_binary_reward_fn():
    def reward_fn(completions, **kwargs):
        rewards = []
        for completion in completions:
            text = completion if isinstance(completion, str) else completion[0].get("content", "")
            has_answer = "[ANSWER]" in text
            rewards.append(1.0 if has_answer else 0.0)
        return rewards
    return reward_fn

def make_toolrl_reward_fn():
    def reward_fn(completions, **kwargs):
        prompts = kwargs.get("prompts", [""] * len(completions))
        rewards = []
        for completion in completions:
            tool_calls = re.findall(r"\[TOOL_CALL\]\s+(\w+)\((.*?)\)", completion, re.DOTALL)
            has_answer = "[ANSWER]" in completion

            if not tool_calls:
                rewards.append(0.0)
                continue

            name_scores = []
            schema_scores = []
            value_scores = []
            for name, args_str in tool_calls:
                # Tool name validity
                bad = ["placeholder", "tool_name", "example"]
                name_scores.append(0.0 if any(b in name.lower() for b in bad) else 1.0)
                # Schema: try parse args
                try:
                    args = json.loads(args_str)
                    schema_scores.append(1.0 if isinstance(args, dict) and args else 0.0)
                    non_empty = sum(1 for v in args.values() if v not in (None, "", [], {}))
                    value_scores.append(non_empty / max(len(args), 1))
                except Exception:
                    schema_scores.append(0.0)
                    value_scores.append(0.0)

            r = (
                0.4 * (1.0 if has_answer else 0.0)
                + 0.2 * np.mean(name_scores)
                + 0.2 * np.mean(schema_scores)
                + 0.2 * np.mean(value_scores)
            )
            rewards.append(float(r))
        return rewards
    return reward_fn

def make_irl_reward_fn(theta):
    # IRL reward applied to completions by parsing them back into a pseudo-trajectory and computing θᵀφ.
    sys.path.insert(0, str(Path(__file__).parent))

    def parse_completion_to_pseudo_traj(prompt: str, completion: str) -> dict:
        # Parse a model completion into a pseudo-trajectory for feature computation.
        query_match = re.search(r"\[QUERY\]\s*(.*?)(?:\n|$)", prompt)
        query = query_match.group(1).strip() if query_match else ""
        steps = []
        step_idx = 0

        # Parse TOOL_CALL blocks & reject obvious hallucinations
        for m in re.finditer(r"\[TOOL_CALL\]\s+(\w+)\((.+?)\)(?=\s*\[|\s*$)", completion, re.DOTALL):
            name = m.group(1).strip()
            if any(bad in name.lower() for bad in ["tool_name", "function_name", "placeholder"]):
                continue
            try:
                args = json.loads(m.group(2).strip())
                if not isinstance(args, dict):
                    args = {}
            except Exception:
                args = {}
                for kv in re.finditer(r'"(\w+)"\s*:\s*"([^"]*)"', m.group(2)):
                    args[kv.group(1)] = kv.group(2)
            steps.append({
                "step_idx": step_idx, "role": "assistant",
                "tool_name": name, "tool_args": args,
                "tool_output": None, "raw_content": m.group(0),
            })
            step_idx += 1

        # Parse TOOL_RESULT blocks
        for m in re.finditer(r"\[TOOL_RESULT\]\s*(.*?)(?=\[TOOL_CALL\]|\[ANSWER\]|$)", completion, re.DOTALL):
            output = m.group(1).strip()[:300]
            if output:
                steps.append({
                    "step_idx": step_idx, "role": "tool",
                    "tool_name": None, "tool_args": None,
                    "tool_output": output, "raw_content": output,
                })
                step_idx += 1

        answer_match = re.search(r"\[ANSWER\]\s*(.*?)$", completion, re.DOTALL)
        final_answer = answer_match.group(1).strip() if answer_match else None
        tool_names_used = [s["tool_name"] for s in steps
                           if s["role"] == "assistant" and s["tool_name"]]
        return {
            "id": "generated", "query": query, "domain": "", "api": "",
            "pass_rate": 0.0, "steps": steps, "final_answer": final_answer,
            "finish_type": "give_answer" if final_answer else "give_up_and_restart",
            "n_tool_calls": len([s for s in steps if s["role"] == "assistant" and s["tool_name"]]),
            "tool_names_used": list(dict.fromkeys(tool_names_used)),
        }


    _spec2 = spec_from_file_location('feature_extraction', Path(__file__).parent / 'feature_extraction.py')
    _mod2 = _mfs2(_spec2); _spec2.loader.exec_module(_mod2)
    compute_features = _mod2.compute_features

    def reward_fn(completions, **kwargs):
        prompts = kwargs.get('prompts', [''] * len(completions))
        rewards = []
        for prompt, completion in zip(prompts, completions):
            completion = completion if isinstance(completion, str) else completion[0].get('content', '')
            traj = parse_completion_to_pseudo_traj(prompt, completion)
            features = np.array(compute_features(traj))
            raw = float(theta @ features)
            # Normalise to [0, 1]
            normalised = (np.clip(raw, IRL_REWARD_MIN, IRL_REWARD_MAX) - IRL_REWARD_MIN) / (IRL_REWARD_MAX - IRL_REWARD_MIN)
            rewards.append(normalised)
        return rewards

    return reward_fn

CONDITION_CHOICES = ["binary", "toolrl", "irl"]
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITION_CHOICES, default="binary", help="Which reward condition to train")
    args = parser.parse_args()
    condition = args.condition

    output_dir = MODEL_BASE / condition
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"GRPO {condition} reward training\n")

    # Load data
    with open(DATA_DIR / "expert_parsed.json") as f:
        expert = json.load(f)

    prompts = [format_prompt(t) for t in expert]
    print(f"Training examples: {len(prompts)}")

    has_gpu = torch.cuda.is_available()
    if has_gpu:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb < 6:
            print(f"Insufficient VRAM ({vram_gb:.1f}GB) — saving config for AMD cloud")
            has_gpu = False

    if not has_gpu:
        print("No GPU available")
        config = {
            "condition": condition,
            "base_model": BASE_MODEL,
            "grpo_args": GRPO_ARGS,
            "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]},
            "n_train_examples": len(prompts),
            "output_dir": str(output_dir),
        }
        with open(output_dir / "training_config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save prompt dataset
        dataset_path = DATA_DIR / f"grpo_dataset_{condition}.json"
        with open(dataset_path, "w") as f:
            json.dump([{"prompt": p, "traj_id": t["id"]} for p, t in zip(prompts, expert)], f)
        print(f"Config saved == {output_dir}/training_config.json")
        print(f"Dataset saved == {dataset_path}")
        return

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    # Set up reward function
    if condition == "binary":
        reward_fn = make_binary_reward_fn()
    elif condition == "toolrl":
        reward_fn = make_toolrl_reward_fn()
    else: # irl
        with open(Path("output") / "theta_weights.json") as f:
            theta_data = json.load(f)
        theta = np.array(theta_data["theta"])
        reward_fn = make_irl_reward_fn(theta)
    dataset = Dataset.from_list([{"prompt": p} for p in prompts])

    grpo_args_clean = {k: v for k, v in GRPO_ARGS.items() if k not in ("output_dir", "max_new_tokens")}
    grpo_config = GRPOConfig(
        **grpo_args_clean,
        output_dir=str(output_dir),
        max_completion_length=GRPO_ARGS.get("max_new_tokens", 512),
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=dataset,
        peft_config=lora_config,
    )

    print(f"\nStarting GRPO training ({condition} reward)...")
    trainer.train()
    trainer.save_model(str(output_dir))
    print(f"\nModel saved == {output_dir}")

    log = [
        {"step": e["step"], "loss": e.get("loss"), "reward": e.get("reward")}
        for e in trainer.state.log_history
    ]
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"Log saved == {output_dir}/training_log.json")


if __name__ == "__main__":
    main()