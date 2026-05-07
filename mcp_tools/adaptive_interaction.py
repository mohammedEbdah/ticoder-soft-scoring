"""
mcp_tools/adaptive_interaction.py
Tool 1: entropy-based adaptive stopping for Phase 3.

Stop criterion from CONTEXT.md:
- Group surviving codes by behavioral signature.
- Compute H = -sum(p_i * log2(p_i)).
- If H < 0.5, stop interactions early.
"""

import math
import os
import sys
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_tools.rank_by_clustering import cluster_codes

DEFAULT_ENTROPY_THRESHOLD = 0.5
MIN_INTERACTIONS = 1


def compute_entropy(codes: List[str], approved_tests: List[str]) -> float:
    """
    Compute Shannon entropy over behavioral clusters.

    Args:
        codes: Current surviving code suggestions
        approved_tests: Oracle-approved tests used to define behavior signatures

    Returns:
        Entropy value.
    """
    if not codes:
        return 0.0

    # With no approved tests yet, treat each code as its own behavior.
    if not approved_tests:
        return math.log2(len(codes)) if len(codes) > 1 else 0.0

    clusters = cluster_codes(codes, approved_tests)
    n_total = len(codes)
    entropy = 0.0

    for indices in clusters.values():
        p_i = len(indices) / n_total
        if p_i > 0:
            entropy -= p_i * math.log2(p_i)

    return entropy


def should_stop(
    codes: List[str],
    approved_tests: List[str],
    remaining_tests: List[str],
    current_m: int,
    original_n_codes: int,
    entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
    min_interactions: int = MIN_INTERACTIONS,
) -> Tuple[bool, str]:
    """
    Decide whether to stop the interaction loop early.

    Stops when:
    - No remaining tests
    - Only one code survives (entropy edge case H=0)
    - Entropy H < threshold
    """
    del original_n_codes  # Kept for compatibility with existing call sites.

    if current_m < min_interactions:
        return False, f"Below minimum interactions ({min_interactions})"

    if not remaining_tests:
        return True, "No remaining tests to show"

    if len(codes) <= 1:
        return True, f"Only {len(codes)} code(s) remaining"

    entropy = compute_entropy(codes, approved_tests)
    if entropy < entropy_threshold:
        return True, f"Entropy ({entropy:.3f}) below threshold ({entropy_threshold})"

    return False, f"Entropy={entropy:.3f} >= threshold={entropy_threshold}, continuing"


def get_interaction_stats(
    codes_per_m: Dict[int, List[str]],
    approved_tests_per_m: Dict[int, List[str]],
    original_n_codes: int,
) -> List[dict]:
    """Compute per-interaction entropy stats for analysis."""
    stats: List[dict] = []

    for m in sorted(codes_per_m.keys()):
        codes = codes_per_m[m]
        approved = approved_tests_per_m.get(m, [])
        entropy = compute_entropy(codes, approved)

        stop, reason = should_stop(
            codes,
            approved,
            remaining_tests=["placeholder"],
            current_m=m,
            original_n_codes=original_n_codes,
        )

        stats.append(
            {
                "m": m,
                "n_surviving": len(codes),
                "entropy": round(entropy, 3),
                "would_stop": stop,
                "stop_reason": reason,
            }
        )

    return stats


def compute_adaptive_m(
    codes_history: Dict[int, List[str]],
    approved_tests_history: Dict[int, List[str]],
    original_n_codes: int,
    entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
    min_interactions: int = MIN_INTERACTIONS,
) -> int:
    """Compute first interaction m where adaptive stopping would trigger."""
    for m in sorted(codes_history.keys()):
        stop, _ = should_stop(
            codes_history[m],
            approved_tests_history.get(m, []),
            remaining_tests=["placeholder"],
            current_m=m,
            original_n_codes=original_n_codes,
            entropy_threshold=entropy_threshold,
            min_interactions=min_interactions,
        )
        if stop:
            return m

    return max(codes_history.keys()) if codes_history else 5
