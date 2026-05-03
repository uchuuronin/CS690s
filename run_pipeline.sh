#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

GPU_MODE=false
CONDITION="all"
SKIP_DOWNLOAD=false; SKIP_PARSE=false; SKIP_FEATURES=false; SKIP_IRL=false; SKIP_SFT=false
ONLY_IRL=false
IRL_LR=0.05; IRL_L2=0.01; IRL_ITERS=500; IRL_PAIR_DELTA=0.15
N_EXPERT=300; N_HELD_OUT=80; PASS_RATE=0.8
DATA_ROOT="data"; GROUP="G1"
REWARD_SOURCE="maxent"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU_MODE=true ;;
    --condition) CONDITION="$2"; shift ;;
    --skip-download) SKIP_DOWNLOAD=true ;;
    --skip-parse) SKIP_PARSE=true ;;
    --skip-features) SKIP_FEATURES=true ;;
    --skip-irl) SKIP_IRL=true ;;
    --skip-sft) SKIP_SFT=true ;;
    --only-irl) ONLY_IRL=true ;;
    --lr) IRL_LR="$2"; shift ;;
    --l2) IRL_L2="$2"; shift ;;
    --iters) IRL_ITERS="$2"; shift ;;
    --pair-delta) IRL_PAIR_DELTA="$2"; shift ;;
    --n-expert) N_EXPERT="$2"; shift ;;
    --n-held-out) N_HELD_OUT="$2"; shift ;;
    --pass-rate) PASS_RATE="$2"; shift ;;
    --data-root) DATA_ROOT="$2"; shift ;;
    --group) GROUP="$2"; shift ;;
    --reward-source) REWARD_SOURCE="$2"; shift ;;
    --help|-h) grep "^#" "$0" | head -30 | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown flag: $1"; exit 1 ;;
  esac
  shift
done

$ONLY_IRL && GPU_MODE=false
mkdir -p data output models

pip install datasets tqdm numpy scipy --break-system-packages -q
$GPU_MODE && pip install trl peft transformers accelerate bitsandbytes torch \
  --extra-index-url https://download.pytorch.org/whl/rocm6.1 --break-system-packages -q

if $SKIP_DOWNLOAD; then
  echo "[1] skipping download"
else
  echo "[1] filtering ToolBench..."
  [ ! -d "$DATA_ROOT/answer" ] && echo "ERROR: $DATA_ROOT/answer not found" && exit 1
  python pipeline/preprocessor.py \
    --data-root "$DATA_ROOT" --group "$GROUP" \
    --n-expert "$N_EXPERT" --n-held-out "$N_HELD_OUT" --pass-rate "$PASS_RATE"
fi

$SKIP_PARSE && echo "[2] skipping parse" || { echo "[2] parsing trajectories..."; python pipeline/parse_trajectories.py; }
$SKIP_FEATURES && echo "[3] skipping features" || { echo "[3] extracting features..."; python pipeline/feature_extraction.py; }
$SKIP_IRL && echo "[4] skipping IRL" || {
  echo "[4] running MaxEnt IRL..."
  python pipeline/maxent_irl.py --lr "$IRL_LR" --l2 "$IRL_L2" --iters "$IRL_ITERS" --pair-delta "$IRL_PAIR_DELTA" --reward-source "$REWARD_SOURCE"
}

echo "[5] reward stats..."
python pipeline/reward_functions.py

$ONLY_IRL && echo "done (--only-irl)" && exit 0

$SKIP_SFT && echo "[6] skipping SFT" || { echo "[6] SFT baseline..."; python pipeline/sft_baseline.py; }

echo "[7] GRPO training (condition=$CONDITION)..."
CONDITIONS=$( [[ "$CONDITION" == "all" ]] && echo "binary toolrl irl" || echo "$CONDITION" )
for C in $CONDITIONS; do
  echo "  $C"
  python pipeline/grpo_train.py --condition "$C"
done

echo "[8] feature ablation and theta comparison..."
python pipeline/ablation.py

echo "[9] evaluation: inference + Mann-Whitney H2..."
python pipeline/analysis.py

echo "done."
