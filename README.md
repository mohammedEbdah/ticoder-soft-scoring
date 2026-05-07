# Extending TICODER with Soft Scoring and Entropy-Based Early Stopping

A reproduction and extension of [TICODER: Test-driven Interactive Code Generation](https://arxiv.org/abs/2210.15187) on the MBPP benchmark using Claude Haiku.

## Overview

TICODER is a test-driven interactive code generation framework that uses oracle (user/developer) feedback to select the best code suggestion from a pool of LLM-generated candidates. The original paper uses **hard pruning** — permanently removing codes that disagree with the oracle.

We extend TICODER with two tightly coupled improvements that form a **unified feedback loop**:

1. **Soft Scoring** — Replace hard pruning with cumulative +1/-1 scoring. No code is ever permanently removed, making the system resilient to oracle mistakes.
2. **Entropy-Based Early Stopping** — After each oracle interaction, cluster codes by behavioral signature and compute Shannon entropy. Stop when entropy drops below 0.5 (one behavioral cluster dominates). This is only meaningful with soft scoring — under hard pruning, entropy is trivially zero.

## Key Results (MBPP, 427 tasks)

| Configuration | pass@1@5 |
|---|---|
| Baseline (random pick) | 54.05% |
| TICODER Hard Pruning (reproduction) | 67.92% |
| **Soft Scoring + Entropy Stop (ours)** | **69.56%** |

- **+1.63 points** over hard pruning
- **54% fewer oracle calls** (avg m=2.3 instead of 5)
- **80% of tasks** stop early
- Under 20% oracle noise: soft scoring loses only 1.17 points vs 6.09 for hard pruning

## Project Structure

```
ticoder_v3/
├── config.py                    # Central configuration (model, N, T, temperature)
├── requirements.txt             # Python dependencies
├── data/
│   ├── download_datasets.py     # MBPP download and loading
│   └── mbpp.jsonl               # MBPP dataset (427 sanitized tasks)
├── utils/
│   ├── prompts.py               # LLM prompt templates (code + test generation)
│   ├── oracle.py                # Oracle simulation (reference solution-based)
│   ├── metrics.py               # pass@k, pass@k@m, early stopping stats
│   └── safe_io.py               # Atomic JSON I/O with crash recovery
├── ticoder_core/
│   ├── generator.py             # LLM code/test generation (N=100, T=50)
│   ├── ranker.py                # s_discr discriminative test ranking
│   ├── pruner.py                # Hard pruning (original TICODER)
│   └── workflow.py              # Main TICODER interaction loop
├── mcp_tools/
│   ├── soft_scoring.py          # Soft scoring (+1/-1 accumulation)
│   ├── rank_by_clustering.py    # Behavioral clustering for code ranking
│   ├── adaptive_interaction.py  # Entropy-based early stopping
│   └── noise_recovery.py        # Noisy oracle simulation (Phase 4)
├── experiments/
│   └── run_all.py               # Main entry point for all 4 phases
├── results/                     # Experiment output JSON files
└── paper/
    └── main.tex                 # IEEE conference paper (LaTeX)
```

## Experiment Settings

| Parameter | Value |
|---|---|
| Model | Claude Haiku (`claude-haiku-4-5`) |
| Dataset | MBPP — 427 tasks (sanitized split) |
| Code suggestions (N) | 100 per task |
| Test suggestions (T) | 50 per task |
| Temperature | 0.8 |
| Max interactions (m) | 5 |
| Oracle | Reference solution (simulated perfect oracle) |
| Entropy threshold | 0.5 |

## How It Works

### The Feedback Loop

```
┌─────────────────────────────────────────────────────────┐
│  1. s_discr picks the most informative test             │
│  2. Oracle answers PASS or FAIL                         │
│  3. Soft scoring accumulates evidence (+1/-1)           │
│  4. Entropy checks: is the ranking settled?             │
│     → If H < 0.5: STOP (dominant cluster found)        │
│     → If H ≥ 0.5: go back to step 1                   │
└─────────────────────────────────────────────────────────┘
```

### Why Entropy Needs Soft Scoring

Under hard pruning, all surviving codes agree on every approved test by construction (disagreeing codes were already eliminated). This means behavioral signatures are all identical, entropy is trivially zero, and the stopping criterion fires after the first approval — providing no useful signal.

Soft scoring keeps all codes alive, so codes *can* disagree on approved tests. Entropy then reflects genuine behavioral diversity, making it a meaningful stopping criterion.

## Four Experimental Phases

1. **Pilot** (20 tasks) — Validate implementation correctness
2. **Reproduction** (427 tasks) — Reproduce TICODER with hard pruning, establish baselines
3. **Extension** (427 tasks) — Evaluate soft scoring + entropy stopping vs hard pruning
4. **Noise Test** (427 tasks) — Inject 1 random oracle mistake per task (20% noise), compare robustness

Phases 3 and 4 reuse cached LLM generations from Phase 2 — no additional API calls.

## Running the Experiments

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key (required for Phases 1-2 only)
export ANTHROPIC_API_KEY=sk-ant-...

# Run phases sequentially
python experiments/run_all.py --phase pilot      # Phase 1: quick validation
python experiments/run_all.py --phase reproduce   # Phase 2: full reproduction
python experiments/run_all.py --phase mcp         # Phase 3: our extensions
python experiments/run_all.py --phase noisy       # Phase 4: noise robustness
```

Phases 3 and 4 require Phase 2 cache to exist. They will fail fast if cache is missing.

## Cost Reference

| Mode | Phase 1 (20 tasks) | Phase 2 (427 tasks) | Phases 3-4 | Total |
|---|---|---|---|---|
| Standard API | ~$0.94 | ~$20 | $0 (cached) | ~$21 |
| Batch API | ~$0.47 | ~$10 | $0 (cached) | ~$11 |

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{ticoder_extension_2026,
  title={Extending TICODER with Soft Scoring and Entropy-Based Early Stopping for Robust Interactive Code Generation},
  author={Noori},
  year={2026}
}
```

## Acknowledgments

- Original TICODER paper: [Fakhoury et al., 2024](https://arxiv.org/abs/2210.15187)
- Original TICODER code: [github.com/microsoft/ticoder](https://github.com/microsoft/ticoder)
- MBPP benchmark: [Austin et al., 2021](https://arxiv.org/abs/2108.07732)
