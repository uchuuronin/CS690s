"""
About:
    Uses LoRA via TRL's SFTTrainer. Trains on the query == tool_call sequence as a language modelling objective.
"""
import json
import os
from pathlib import Path
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer
from trl import SFTConfig
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, OUTPUT_DIR, MODELS_DIR, BASE_MODEL, MAX_SEQ_LEN, LORA_R, 
    LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES, SFT_ARGS, 
)
MODEL_DIR = MODELS_DIR / "sft"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_ARGS = {**SFT_ARGS, "output_dir": str(MODEL_DIR)}


def format_trajectory_for_sft(traj: dict) -> str:
    parts = [f"[QUERY] {traj['query']}"]
    for step in traj["steps"]:
        if step["role"] == "assistant" and step["tool_name"]:
            args_str = json.dumps(step["tool_args"] or {})
            parts.append(f"[TOOL_CALL] {step['tool_name']}({args_str})")
        elif step["role"] == "tool":
            out = (step["tool_output"] or "")[:200]
            parts.append(f"[TOOL_RESULT] {out}")
    if traj.get("final_answer"):
        parts.append(f"[ANSWER] {traj['final_answer'][:500]}")
    return "\n".join(parts)


def main():
    # Load parsed trajectories
    with open(DATA_DIR / "expert_parsed.json") as f:
        expert = json.load(f)

    texts = [format_trajectory_for_sft(t) for t in expert]
    print(f"Training examples: {len(texts)}")
    print(f"\nExample (first 300 chars):\n{texts[0][:300]}\n")

    # Check if we're in a compute environment with GPU
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        if vram_gb < 6:
            print(f"Insufficient VRAM ({vram_gb:.1f}GB < 15GB)")
            has_gpu = False
    if not has_gpu:
        print("No GPU detected")
        print("Saving training config for AMD cloud run...")
        config = {
            "base_model": BASE_MODEL,
            "train_args": TRAIN_ARGS,
            "lora": {
                "r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            },
            "n_train_examples": len(texts),
            "max_seq_len": MAX_SEQ_LEN,
            "format": "query == tool_calls == answer",
        }
        with open(MODEL_DIR / "training_config.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"Config saved == {MODEL_DIR}/training_config.json")

        # Save formatted dataset for AMD cloud
        dataset_path = DATA_DIR / "sft_dataset.json"
        with open(dataset_path, "w") as f:
            json.dump([{"text": t} for t in texts], f, indent=2)
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
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    dataset = Dataset.from_list([{"text": t} for t in texts])

    train_args_clean = {k: v for k, v in TRAIN_ARGS.items() if k != "output_dir"}
    sft_args = SFTConfig(
        **train_args_clean,
        output_dir=str(MODEL_DIR),
        max_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        completion_only_loss=False,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        peft_config=lora_config,
        args=sft_args,
    )

    print("\nStarting SFT training...")
    trainer.train()
    trainer.save_model(str(MODEL_DIR))
    print(f"\nSFT model saved == {MODEL_DIR}")

    # Save training log
    log = [
        {"step": entry["step"], "loss": entry["loss"]}
        for entry in trainer.state.log_history
        if "loss" in entry
    ]
    with open(MODEL_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"Training log saved == {MODEL_DIR}/training_log.json")


if __name__ == "__main__":
    main()