#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — MaxEnt IRL Tool-Use Rewards Pipeline
#
# Must be run from the project root (where this file lives).
#
# USAGE
#   bash run_pipeline.sh [FLAGS]
#
# FLAGS
#   --gpu               Enable GPU-dependent stages (SFT + GRPO training).
#                       Without this flag, stages 1–4 run fully; training
#                       stages save configs/datasets for AMD cloud and exit.
#
#   --condition <n>     Which GRPO reward condition to train. Only used when
#                       --gpu is set. Options:
#                         binary   — pass/fail outcome only
#                         toolrl   — hand-crafted ToolRL-style decomposed reward
#                         irl      — IRL-learned reward (θᵀφ, normalised)
#                         all      — run all three sequentially (default)
#
#   --skip-download     Skip Stage 1 (download + filter). Use if data/
#                       already contains expert_trajectories.json,
#                       held_out_trajectories.json, and
#                       suboptimal_trajectories.json from a previous run.
#
#   --skip-parse        Skip Stage 2 (parse trajectories). Use if
#                       expert_parsed.json, held_out_parsed.json, and
#                       suboptimal_parsed.json already exist.
#
#   --skip-features     Skip Stage 3 (feature extraction). Use if
#                       expert_features.json, held_out_features.json, and
#                       suboptimal_features.json already exist.
#
#   --skip-irl          Skip Stage 4 (MaxEnt IRL). Use if
#                       output/theta_weights.json already exists.
#
#   --skip-sft          Skip SFT baseline training (Stage 6). Only relevant
#                       with --gpu.
#
#   --only-irl          Run stages 1–4 only (data == features == IRL).
#                       Equivalent to omitting --gpu. Useful for iterating
#                       on reward learning before committing to GPU time.
#
#   --lr <float>        Learning rate for MaxEnt IRL gradient ascent.
#                       Default: 0.05
#
#   --l2 <float>        L2 regularisation coefficient for IRL.
#                       Default: 0.01
#
#   --iters <int>       Number of IRL gradient ascent iterations.
#                       Default: 500
#
#   --pair-delta <float> Minimum pass_rate gap between two held-out
#                       trajectories for them to count as a "clear" pair in
#                       the pairwise ranking sanity check.
#                       Default: 0.15
#
#   --n-expert <int>    Number of expert trajectories to collect.
#                       Default: 300
#
#   --n-held-out <int>  Number of held-out trajectories for sanity check.
#                       Default: 80
#
#   --pass-rate <float> Minimum ToolEval pass rate for expert filtering.
#                       Trajectories below this threshold are collected as
#                       the suboptimal set (model distribution for IRL).
#                       Default: 0.8
#
# EXAMPLES
#   # Local run (no GPU): download, features, IRL — save training configs
#   bash run_pipeline.sh
#
#   # Re-run IRL with different hyperparameters, skip slow stages
#   bash run_pipeline.sh --skip-download --skip-parse --skip-features \
#                        --lr 0.01 --l2 0.001 --iters 1000
#
#   # Re-run IRL sanity check with tighter pair threshold
#   bash run_pipeline.sh --skip-download --skip-parse --skip-features \
#                        --pair-delta 0.2
#
#   # Full AMD cloud run: all stages including all three GRPO conditions
#   bash run_pipeline.sh --gpu --condition all
#
#   # AMD cloud: train only IRL condition (others already done)
#   bash run_pipeline.sh --gpu --condition irl \
#                        --skip-download --skip-parse --skip-features \
#                        --skip-irl --skip-sft
#
# DIRECTORY STRUCTURE (all relative to project root)
#   data/       Downloaded + parsed trajectories (expert, held-out, suboptimal)
#   pipeline/   Python stage scripts
#   output/     Results: theta_weights.json, feature_report.json, etc.
#   models/     Trained LoRA adapters (created only with --gpu)
#
# OUTPUT FILES
#   output/filter_report.json     Trajectory filtering statistics
#   output/feature_report.json    Per-feature stats + ToolEval correlation
#   output/theta_weights.json     Recovered θ weights  ← main IRL result
#   output/irl_training_log.json  Log-likelihood + μ gap per iteration
#   output/sanity_check.json      Pairwise ranking accuracy on held-out
#   models/sft/                   SFT baseline adapter        (--gpu)
#   models/binary/                GRPO binary reward adapter   (--gpu)
#   models/toolrl/                GRPO ToolRL reward adapter   (--gpu)
#   models/irl/                   GRPO IRL reward adapter      (--gpu)
# =============================================================================

set -e
cd "$(dirname "$0")"

# ── Defaults ──────────────────────────────────────────────────────────────────
GPU_MODE=false
CONDITION="all"
SKIP_DOWNLOAD=false
SKIP_PARSE=false
SKIP_FEATURES=false
SKIP_IRL=false
SKIP_SFT=false
ONLY_IRL=false
IRL_LR=0.05
IRL_L2=0.01
IRL_ITERS=500
IRL_PAIR_DELTA=0.15
N_EXPERT=300
N_HELD_OUT=80
PASS_RATE=0.8
DATA_ROOT="data"
GROUP="G1"
FULL_DATASET=false

# ── Parse flags ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)            GPU_MODE=true ;;
    --condition)      CONDITION="$2"; shift ;;
    --skip-download)  SKIP_DOWNLOAD=true ;;
    --skip-parse)     SKIP_PARSE=true ;;
    --skip-features)  SKIP_FEATURES=true ;;
    --skip-irl)       SKIP_IRL=true ;;
    --skip-sft)       SKIP_SFT=true ;;
    --only-irl)       ONLY_IRL=true ;;
    --lr)             IRL_LR="$2"; shift ;;
    --l2)             IRL_L2="$2"; shift ;;
    --iters)          IRL_ITERS="$2"; shift ;;
    --pair-delta)     IRL_PAIR_DELTA="$2"; shift ;;
    --n-expert)       N_EXPERT="$2"; shift ;;
    --n-held-out)     N_HELD_OUT="$2"; shift ;;
    --pass-rate)      PASS_RATE="$2"; shift ;;
    --data-root)      DATA_ROOT="$2"; shift ;;
    --group)          GROUP="$2"; shift ;;
    --full-dataset)   FULL_DATASET=true ;;
    --help|-h)        grep "^#" "$0" | head -90 | sed 's/^# \?//'; exit 0 ;;
    *) echo "Unknown flag: $1  (run with --help for usage)"; exit 1 ;;
  esac
  shift
done

if $ONLY_IRL; then GPU_MODE=false; fi

echo "============================================="
echo "  MaxEnt IRL Tool-Use Rewards Pipeline"
echo "============================================="
echo "  GPU mode    : $GPU_MODE"
echo "  Condition   : $CONDITION"
echo "  IRL lr/l2   : $IRL_LR / $IRL_L2  ($IRL_ITERS iters)"
echo "  Pair delta  : $IRL_PAIR_DELTA  (pairwise ranking sanity check)"
echo "  Expert N    : $N_EXPERT  |  Held-out: $N_HELD_OUT"
echo "  Pass rate   : >= $PASS_RATE  (below this == suboptimal set for IRL)"
echo ""

mkdir -p data output models

# ── Install dependencies ──────────────────────────────────────────────────────
echo "[0] Installing dependencies..."
pip install datasets tqdm huggingface_hub numpy scipy --break-system-packages -q
$GPU_MODE && pip install trl peft transformers accelerate bitsandbytes torch --break-system-packages -q

# ── Stage 1: Download + filter ────────────────────────────────────────────────
if $SKIP_DOWNLOAD; then
  echo "[1] Skipping download (--skip-download)"
else
  echo "[1] Downloading and filtering ToolBench..."
  echo "    Collecting: expert (pass_rate>=$PASS_RATE), held-out, and suboptimal (for IRL)"
  if [ ! -d "$DATA_ROOT/answer" ]; then
    echo "ERROR: $DATA_ROOT/answer not found. Run: bash fetch_data.sh first."
    exit 1
  fi
  STAGE1_ARGS="--data-root $DATA_ROOT --group $GROUP \
    --n-expert $N_EXPERT --n-held-out $N_HELD_OUT --pass-rate $PASS_RATE"
  $FULL_DATASET && STAGE1_ARGS="$STAGE1_ARGS --full-dataset"
  python pipeline/preprocessor.py $STAGE1_ARGS
fi

# ── Stage 2: Parse ────────────────────────────────────────────────────────────
if $SKIP_PARSE; then
  echo "[2] Skipping parse (--skip-parse)"
else
  echo "[2] Parsing trajectories (expert + held-out + suboptimal)..."
  python pipeline/parse_trajectories.py
fi

# ── Stage 3: Feature extraction ───────────────────────────────────────────────
if $SKIP_FEATURES; then
  echo "[3] Skipping features (--skip-features)"
else
  echo "[3] Extracting 8-dim φ(τ) feature vectors..."
  python pipeline/03_feature_extraction.py
fi

# ── Stage 4: MaxEnt IRL ───────────────────────────────────────────────────────
if $SKIP_IRL; then
  echo "[4] Skipping IRL (--skip-irl)"
else
  echo "[4] Running MaxEnt IRL..."
  echo "    Model distribution: real suboptimal trajectories (pass_rate < $PASS_RATE)"
  python pipeline/04_maxent_irl.py \
    --lr "$IRL_LR" --l2 "$IRL_L2" --iters "$IRL_ITERS" \
    --pair-delta "$IRL_PAIR_DELTA"
fi

# ── Stage 5: Reward report ────────────────────────────────────────────────────
echo "[5] Reward distribution report..."
python pipeline/06_reward_functions.py

if $ONLY_IRL; then
  echo ""
  echo "Done (--only-irl). Share output/ for midpoint report."
  exit 0
fi

# ── Stage 6: SFT baseline ────────────────────────────────────────────────────
if $SKIP_SFT; then
  echo "[6] Skipping SFT (--skip-sft)"
else
  echo "[6] SFT baseline..."
  python pipeline/05_sft_baseline.py
fi

# ── Stage 7: GRPO training ───────────────────────────────────────────────────
echo "[7] GRPO training (condition: $CONDITION)..."
CONDITIONS=$( [[ "$CONDITION" == "all" ]] && echo "binary toolrl irl" || echo "$CONDITION" )
for C in $CONDITIONS; do
  echo "  == $C"
  python pipeline/07_grpo_train.py --condition "$C"
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================="
echo "  Done."
echo "============================================="
echo "  output/filter_report.json"
echo "  output/feature_report.json"
echo "  output/theta_weights.json       ← main IRL result"
echo "  output/irl_training_log.json    ← log-likelihood + μ gap per iter"
echo "  output/sanity_check.json        ← pairwise ranking accuracy"
$GPU_MODE && echo "  models/{sft,binary,toolrl,irl}/"
! $GPU_MODE && echo "  models/*/training_config.json   ← transfer to AMD + run with --gpu"
echo ""