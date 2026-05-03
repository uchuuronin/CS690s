# MaxEnt IRL Tool-Use Rewards

CS 690S Final Project — recovering implicit tool-use reward functions via Maximum Entropy IRL from ToolBench expert demonstrations, then training an LLM agent (Llama-3-8B + LoRA) via GRPO against the learned reward.

## Structure

```
run_pipeline.sh       Master runner — start here
README.md

pipeline/
  preprocessor.py         Filter ToolBench into expert / held-out / suboptimal sets
  02_parse_trajectories.py    Parse raw JSON into structured step format (all three splits)
  03_feature_extraction.py    Compute 8-dim φ(τ) feature vectors (all three splits)
  04_maxent_irl.py            MaxEnt IRL — recover θ using real suboptimal trajectories as model dist.
  05_sft_baseline.py          SFT baseline (Llama-3-8B + LoRA)
  06_reward_functions.py      All three reward functions + distribution report
  07_grpo_train.py            GRPO training (--condition binary/toolrl/irl)

data/                         All trajectory files (auto-created by pipeline)
  expert_trajectories.json    Raw expert trajectories (pass_rate >= threshold)
  held_out_trajectories.json  Raw held-out trajectories (varied pass_rate)
  suboptimal_trajectories.json  Real failed/partial ToolBench trajectories (model dist. for IRL)
  *_parsed.json               Structured step-level format
  *_features.json             8-dim φ(τ) vectors

output/                       Results (auto-created)
  filter_report.json          Trajectory counts and filtering stats
  feature_report.json         Per-feature stats + ToolEval correlation
  theta_weights.json          Recovered θ weights  ← main IRL result
  irl_training_log.json       Log-likelihood + feature matching gap per iteration
  sanity_check.json           Pairwise ranking accuracy on held-out

models/                       Trained LoRA adapters (GPU stages only)
  sft/  binary/  toolrl/  irl/
```

## Before you start — download the data

ToolBench must be downloaded manually (automated download is unreliable due to Google Drive quota limits):

1. Go to: https://drive.google.com/drive/folders/1TysbSWYpP8EioFu9xPJtpbJZMLLmwAmL
2. Download `data.zip`
3. Unzip into the project root so the structure is `data/answer/`, `data/instruction/`, `data/toolenv/`, etc.

**Windows (PowerShell):**
```powershell
Expand-Archive -Path "$env:USERPROFILE\Downloads\data.zip" -DestinationPath data
```

**Linux / macOS:**
```bash
unzip ~/Downloads/data.zip -d data
```

Verify: `ls data/` should show `answer/  instruction/  toolenv/  ...`

## Quickstart

**Local (no GPU) — stages 1–4:**
```bash
bash run_pipeline.sh
```

**AMD cloud — full pipeline including training:**
```bash
bash run_pipeline.sh --gpu --condition all
```

**Re-run IRL only with different hyperparameters:**
```bash
bash run_pipeline.sh --skip-download --skip-parse --skip-features \
                     --lr 0.01 --l2 0.001 --iters 1000
```

## All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--gpu` | off | Enable SFT + GRPO training stages |
| `--condition` | `all` | GRPO reward: `binary`, `toolrl`, `irl`, or `all` |
| `--skip-download` | off | Skip Stage 1 |
| `--skip-parse` | off | Skip Stage 2 |
| `--skip-features` | off | Skip Stage 3 |
| `--skip-irl` | off | Skip Stage 4 |
| `--skip-sft` | off | Skip SFT baseline |
| `--only-irl` | off | Stages 1–4 only, no training |
| `--lr` | `0.05` | IRL learning rate |
| `--l2` | `0.01` | IRL L2 regularisation |
| `--iters` | `500` | IRL gradient ascent iterations |
| `--pair-delta` | `0.15` | Min pass_rate gap to count as a clear ranking pair in sanity check |
| `--n-expert` | `300` | Expert trajectory count |
| `--n-held-out` | `80` | Held-out trajectory count |
| `--pass-rate` | `0.8` | Expert quality threshold; trajectories below this → suboptimal set |

## IRL design notes

### Model distribution
MaxEnt IRL requires a contrast between expert demonstrations and a model
distribution p(τ|θ). Stage 1 collects real ToolBench trajectories that fell
below the pass_rate threshold — genuine failed or partial agent behaviour from
ToolBench's DFSDT search tree. These form the model distribution, not synthetic
perturbations. Target: ~2× expert count (~600 trajectories).

### Gradient
At each iteration: ∇L = μ_E − μ_θ − 2λθ  
where μ_E is the fixed expert feature expectation and μ_θ is recomputed each
step from the softmax distribution over all 900 trajectories.

### Sanity check
Pairwise ranking accuracy on held-out: for each pair (τᵢ, τⱼ) where
pass_rate(τᵢ) − pass_rate(τⱼ) > `--pair-delta`, check whether θᵀφ(τᵢ) > θᵀφ(τⱼ).
Random baseline = 0.50. Target ≥ 0.65.

## Features (φ vector, dim=8)

| # | Name | Type | What it measures |
|---|------|------|-----------------|
| 0 | `tool_selection_accuracy` | Explicit | Tool name relevant to query |
| 1 | `arg_correctness` | Explicit | Args non-empty and well-formed |
| 2 | `call_ordering` | **Implicit** | Info-gather before act |
| 3 | `redundancy_avoidance` | **Implicit** | No repeated identical calls |
| 4 | `constraint_adherence` | **Implicit** | Query constraints reflected in args |
| 5 | `info_sufficiency` | **Implicit** | Enough info before final answer |
| 6 | `invasiveness_minimisation` | **Implicit** | No unnecessary sensitive data |
| 7 | `efficiency` | **Implicit** | Appropriate number of calls |

## Key outputs to share for midpoint report

- `output/theta_weights.json` — θ weights, μ gap at convergence, hypothesis check
- `output/feature_report.json` — per-feature stats and ToolEval correlation
- `output/sanity_check.json` — pairwise ranking accuracy (target ≥ 0.65)
- `output/irl_training_log.json` — convergence: log-likelihood and |μ_E − μ_θ| per iter

**What to check on real ToolBench data:**
1. θ: ≥ 2 implicit features (indices 2–7) with |θ| > 0.05
2. Feature matching gap |μ_E − μ_θ| < 0.05 at convergence
3. Pairwise ranking accuracy > 0.60 on held-out pairs
4. IRL reward not perfectly correlated with ToolRL reward (ρ < 0.95 in reward report)