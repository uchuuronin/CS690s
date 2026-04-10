# config for all pipeline knobs, paths, and constants.
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
MODELS_DIR = Path("models")

PASS_RATE_THRESHOLD = 0.8 # give_answer→1.0 / give_up→0.0; split point
TARGET_EXPERT = 300
TARGET_HELD_OUT = 80
TARGET_SUBOPTIMAL = TARGET_EXPERT * 2 # model distribution for IRL

FEATURE_NAMES = [
 "tool_selection_accuracy", # explicit
 "arg_correctness", # explicit
 "call_ordering", # implicit
 "redundancy_avoidance", # implicit
 "constraint_adherence", # implicit
 "info_sufficiency", # implicit
 "invasiveness_minimisation", # implicit
 "efficiency", # implicit
]
N_FEATURES = len(FEATURE_NAMES)
