# MaxEnt IRL Tool-Use Rewards

CS 690S Final Project. Recovers implicit tool-use reward functions via Maximum Entropy IRL from ToolBench expert demonstrations, then trains an LLM agent via GRPO against the learned reward.

## Structure

```
run_pipeline.sh

pipeline/
  preprocessor.py        filter ToolBench into expert / held-out / suboptimal sets
  parse_trajectories.py  parse raw JSON into structured step format
  feature_extraction.py  compute 11-dim feature vectors (8 quality + 3 decoy)
  maxent_irl.py          MaxEnt IRL and BT IRL, recover theta
  sft_baseline.py        SFT baseline (LoRA)
  reward_functions.py    binary / ToolRL / IRL reward functions
  grpo_train.py          GRPO training (--condition binary/toolrl/irl)
  config.py              all knobs

data/
  expert_trajectories.json
  held_out_trajectories.json
  suboptimal_trajectories.json
  *_parsed.json
  *_features.json

output/
  feature_report.json
  theta_maxent.json           MaxEnt IRL theta weights
  theta_bt.json               BT IRL theta weights
  theta_weights.json          canonical theta for GRPO
  theta_comparison.json       Spearman rho between MaxEnt and BT theta
  irl_training_log_maxent.json
  irl_training_log_bt.json
  sanity_check.json
  reward_stats.json

models/
  sft/  binary/  toolrl/  irl/
```

## Data

ToolBench must be downloaded manually. Go to https://drive.google.com/drive/folders/1TysbSWYpP8EioFu9xPJtpbJZMLLmwAmL, download `data.zip`, and unzip into the project root so `data/answer/`, `data/instruction/`, and `data/toolenv/` exist.

## Quickstart

```bash
# stages 1-4 only, no GPU needed
bash run_pipeline.sh --only-irl

# full pipeline
bash run_pipeline.sh --gpu --condition all

# re-run IRL with different hyperparameters
bash run_pipeline.sh --skip-download --skip-parse --skip-features \
                     --lr 0.01 --l2 0.001 --iters 1000

# use BT theta for GRPO instead of MaxEnt
bash run_pipeline.sh --gpu --reward-source bt
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--gpu` | off | enable SFT + GRPO training |
| `--condition` | `all` | GRPO reward: `binary`, `toolrl`, `irl`, or `all` |
| `--reward-source` | `maxent` | which theta to write to `theta_weights.json`: `maxent` or `bt` |
| `--skip-download` | off | skip stage 1 |
| `--skip-parse` | off | skip stage 2 |
| `--skip-features` | off | skip stage 3 |
| `--skip-irl` | off | skip stage 4 |
| `--skip-sft` | off | skip SFT baseline |
| `--only-irl` | off | stages 1-4 only, no training |
| `--lr` | `0.05` | IRL learning rate |
| `--l2` | `0.01` | IRL L2 regularisation |
| `--iters` | `500` | IRL gradient ascent iterations |
| `--pair-delta` | `0.15` | min pass_rate gap for ranking sanity check |
| `--n-expert` | `300` | expert trajectory count |
| `--n-held-out` | `80` | held-out trajectory count |
| `--pass-rate` | `0.8` | expert quality threshold; below this goes to suboptimal set |
| `--data-root` | `data/` | path to ToolBench data |
| `--group` | `G1` | G1 (single-tool), G2, or G3 |

## IRL

Both objectives recover a log-linear reward R = theta . phi(tau).

**MaxEnt.** Gradient ascent on log-likelihood of expert trajectories under a Boltzmann distribution over the full pool. The model feature expectation mu_theta is importance-weighted to correct for the 1:2 expert:suboptimal pool imbalance. Gradient: mu_E minus mu_theta_IS minus 2 * lambda * theta.

**BT.** Logistic regression on query-matched (winner, loser) pairs from DFSDT search trees. Equivalent to MaxEnt IRL under Plackett-Luce (Zhu et al., ICML 2023), but uses query-matched pairs rather than a global pool. Gradient: mean over pairs of sigmoid(-delta) * (phi_w minus phi_l) minus 2 * lambda * theta.

Both run every time. `theta_comparison.json` records Spearman rho between the two theta vectors. Features are standardised before IRL and theta is mapped back to original scale after.

## Features

| Index | Name | Type |
|-------|------|------|
| 0 | `tool_selection_accuracy` | explicit |
| 1 | `arg_correctness` | explicit |
| 2 | `tool_diversity` | implicit |
| 3 | `redundancy_avoidance` | implicit |
| 4 | `constraint_adherence` | implicit |
| 5 | `call_success_rate` | implicit |
| 6 | `arg_completeness` | implicit |
| 7 | `efficiency` | implicit |
| 8 | `response_verbosity` | decoy |
| 9 | `unique_tool_count` | decoy |
| 10 | `trajectory_length_raw` | decoy |

Decoys (8-10) have no expected causal link to quality. Large theta on any decoy means unreliable recovery.

## Key outputs

- `theta_maxent.json` / `theta_bt.json`: recovered theta, feature ranking, hypothesis check, sanity check
- `theta_comparison.json`: Spearman rho between MaxEnt and BT theta vectors
- `feature_report.json`: per-feature stats and pass_rate correlation across splits
- `sanity_check.json`: pairwise ranking accuracy on held-out (target 0.65+)
- `reward_stats.json`: reward distributions and inter-signal correlations