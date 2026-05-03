# config for all pipeline knobs, paths, and constants.
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
MODELS_DIR = Path("models")

PASS_RATE_THRESHOLD = 0.8 # give_answer→1.0 / give_up→0.0; split point
TARGET_EXPERT = 300
TARGET_HELD_OUT = 80
TARGET_SUBOPTIMAL = TARGET_EXPERT * 2 #model distribution for IRL

FEATURE_NAMES = [
 "tool_selection_accuracy", # explicit
 "arg_correctness", # explicit
 "tool_diversity", # implicit
 "redundancy_avoidance",# implicit
 "constraint_adherence",# implicit
 "call_success_rate",# implicit 
 "arg_completeness",# implicit
 "efficiency",# implicit
 "response_verbosity",# decoy
 "unique_tool_count",# decoy
 "trajectory_length_raw",# decoy
]
# Indices of real quality features vs decoy features
QUALITY_FEATURE_IDX = list(range(8))
DECOY_FEATURE_IDX= list(range(8, 11))
N_FEATURES = len(FEATURE_NAMES)

IRL_LR = 0.05
TOOLBENCH_GROUP = "G1" # G1=single-tool 
TOOLBENCH_CATEGORIES = [] # empty== all categories
IRL_L2 = 0.01
IRL_ITERS = 500
IRL_LOG_EVERY = 50
IRL_PAIR_DELTA = 0.15 #min pass_rate gap for a held-out pair to count in ranking check

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct" #"meta-llama/Meta-Llama-3-8B-Instruct"
MAX_SEQ_LEN = 1024
LORA_R= 16
LORA_ALPHA= 32
LORA_DROPOUT= 0.05
LORA_TARGET_MODULES= ["q_proj", "v_proj", "k_proj", "o_proj"]

SFT_ARGS = {"num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "warmup_ratio": 0.05,
            "lr_scheduler_type": "cosine",
            "logging_steps": 10,
            "save_steps": 100,
            "bf16": True,
            "report_to": "none"}

GRPO_ARGS = {"num_train_epochs": 2,
             "per_device_train_batch_size": 1,
             "gradient_accumulation_steps": 8,
             "learning_rate": 1e-4,
             "warmup_ratio": 0.05,
             "lr_scheduler_type": "cosine",
             "logging_steps": 10,
             "save_steps": 100,
             "bf16": True,
             "report_to": "none",
             "num_generations": 4,
             "beta": 0.04,
             "max_new_tokens": 512,
}

TOOLRL_W_OUTCOME = 0.4
TOOLRL_W_NAME =0.2
TOOLRL_W_SCHEMA =0.2
TOOLRL_W_VALUES =0.2

IRL_REWARD_MIN = -5.0
IRL_REWARD_MAX = 5.0

DPO_ARGS = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-5,
    "warmup_ratio": 0.05,
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "save_steps": 100,
    "bf16": True,
    "report_to": "none",
    "beta": 0.1, #KL regularisation
}

DPO_PAIR_MIN_MARGIN = 0.1