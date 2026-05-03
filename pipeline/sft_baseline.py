"""
SFT baseline: LoRA fine-tuning on expert trajectories via TRL SFTTrainer.
"""

import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, MODELS_DIR, BASE_MODEL, MAX_SEQ_LEN,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, SFT_ARGS,
)

MODEL_DIR = MODELS_DIR / "sft"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def format_trajectory(traj: dict) -> str:
    parts = [f"[QUERY] {traj['query']}"]
    for step in traj["steps"]:
        if step["role"] == "assistant" and step["tool_name"]:
            parts.append(f"[TOOL_CALL] {step['tool_name']}({json.dumps(step['tool_args'] or {})})")
        elif step["role"] == "tool":
            parts.append(f"[TOOL_RESULT] {(step['tool_output'] or '')[:200]}")
    if traj.get("final_answer"):
        parts.append(f"[ANSWER] {traj['final_answer'][:500]}")
    return "\n".join(parts)


def main():
    with open(DATA_DIR / "expert_parsed.json") as f:
        expert = json.load(f)

    texts = [format_trajectory(t) for t in expert]
    print(f"training examples: {len(texts)}")

    has_gpu = torch.cuda.is_available()
    if has_gpu:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb < 6:
            print(f"insufficient VRAM ({vram_gb:.1f}GB)")
            has_gpu = False

    if not has_gpu:
        print("no GPU — saving config")
        with open(MODEL_DIR / "training_config.json", "w") as f:
            json.dump({
                "base_model": BASE_MODEL, "train_args": SFT_ARGS,
                "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                         "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]},
                "n_train_examples": len(texts), "max_seq_len": MAX_SEQ_LEN,
            }, f, indent=2)
        with open(DATA_DIR / "sft_dataset.json", "w") as f:
            json.dump([{"text": t} for t in texts], f, indent=2)
        return

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map="auto")

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], task_type="CAUSAL_LM",
    )
    sft_config = SFTConfig(
        **{k: v for k, v in SFT_ARGS.items()},
        output_dir=str(MODEL_DIR),
        max_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        completion_only_loss=False,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list([{"text": t} for t in texts]),
        peft_config=lora_config,
        args=sft_config,
    )

    print("starting SFT training...")
    trainer.train()
    trainer.save_model(str(MODEL_DIR))

    log = [{"step": e["step"], "loss": e["loss"]} for e in trainer.state.log_history if "loss" in e]
    with open(MODEL_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"model saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()
