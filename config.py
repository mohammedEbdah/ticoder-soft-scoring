"""
config.py
Central configuration for TICODER v3 experiments.
"""

import os

# API
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    import warnings
    warnings.warn(
        "ANTHROPIC_API_KEY not set. Cache-only phases (mcp, noisy) will work, "
        "but generation phases (pilot, reproduce) will fail.",
        stacklevel=2,
    )

# Model settings (fixed by CONTEXT.md)
REPRODUCTION_MODEL = "claude-haiku-4-5"
GENERATION_MODEL = REPRODUCTION_MODEL

# Generation settings (fixed)
NUM_CODE_SUGGESTIONS = 100
NUM_TEST_SUGGESTIONS = 50
TEMPERATURE = 0.8
MAX_TOKENS = 150
MAX_INTERACTIONS = 5
USE_BATCH_API = True

# Runtime safety: Phase 3/4 are CPU-heavy and each worker may spawn
# subprocesses for candidate execution. Override with TICODER_WORKERS if
# you want to run faster on a stronger machine.
DEFAULT_WORKERS = max(1, min(4, (os.cpu_count() or 2) - 1))
WORKER_COUNT = int(os.environ.get("TICODER_WORKERS", DEFAULT_WORKERS))

# Dataset
DATASET = "mbpp"
MBPP_PATH = "data/mbpp.jsonl"
PILOT_SIZE = 20
FULL_SIZE = None  # None means all 427 MBPP tasks.

# Outputs
RESULTS_DIR = "results"
CACHE_DIR = "cache"

# Variants
VARIANTS = {
    "baseline": "No TICODER interactions",
    "passfail": "TICODER-PASSFAIL",
    "output": "TICODER-OUTPUT",
}
