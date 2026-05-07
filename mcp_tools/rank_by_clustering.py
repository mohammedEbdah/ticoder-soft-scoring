"""
mcp_tools/rank_by_clustering.py
MCP Tool 2: Behavioral clustering-based code ranking.

Inspired by CodeT (Chen et al. 2022), cited in the TICODER paper (Section II).
The paper explicitly says in Section VIII:
    "As part of future work, we plan to explore if our approach may
     benefit from code and test ranking algorithms in CodeT."

This tool implements that future work suggestion.

How it works:
    1. Execute all surviving codes against all approved tests
    2. For each code, build a "behavior signature" = tuple of outputs
       across all test inputs
    3. Group codes with identical behavior signatures into clusters
    4. Rank clusters by size (larger cluster = more codes agree = 
       more likely to be correct behavior)
    5. Within each cluster, rank by number of passing tests (original s_discr)
    6. Return final ranked list

Why this is better than the paper's ranking:
    - Paper ranks by: number of tests each code passes (d_c score)
    - This ranks by: behavioral consensus among all generated codes
    - If 20/30 codes agree on the same behavior, that behavior is
      likely what the user wants — even if they only approved 2 tests

Example:
    Codes: [c1, c2, c3, c4, c5]
    Behaviors:
        c1: (6, 0, True)   ← cluster A (3 codes)
        c2: (6, 0, True)   ← cluster A
        c3: (6, 0, True)   ← cluster A
        c4: (5, 1, False)  ← cluster B (2 codes)
        c5: (5, 1, False)  ← cluster B
    
    Result: cluster A ranked first (more consensus)
    Top-ranked code = representative of cluster A
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Tuple, Any
from collections import defaultdict
from utils.oracle import execute_code, parse_test_assertion


def _make_hashable(value: Any):
    """Convert nested Python outputs into stable hashable signatures."""
    if isinstance(value, dict):
        items = [
            (_make_hashable(k), _make_hashable(v))
            for k, v in value.items()
        ]
        return (
            "dict",
            tuple(sorted(items, key=repr)),
        )
    if isinstance(value, list):
        return ("list", tuple(_make_hashable(item) for item in value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_make_hashable(item) for item in value))
    if isinstance(value, set):
        return (
            "set",
            tuple(sorted((_make_hashable(item) for item in value), key=repr)),
        )

    try:
        hash(value)
        return value
    except TypeError:
        return ("repr", repr(value))


def build_behavior_signature(code: str, tests: List[str]) -> tuple:
    """
    Build a behavior signature for a code suggestion.
    Signature = tuple of actual outputs across all test inputs.

    Args:
        code: Function implementation string
        tests: List of test assertion strings

    Returns:
        Tuple of output values (hashable for grouping)
        Uses special sentinel values for crashes/errors
    """
    CRASH_SENTINEL = "__CRASH__"
    signature = []

    for test in tests:
        input_str, expected = parse_test_assertion(test)

        if input_str is None:
            signature.append(CRASH_SENTINEL)
            continue

        success, actual_output, error = execute_code(code, input_str)

        if not success:
            signature.append(CRASH_SENTINEL)
        else:
            signature.append(_make_hashable(actual_output))

    return tuple(signature)


def cluster_codes(codes: List[str],
                  approved_tests: List[str]) -> Dict[tuple, List[int]]:
    """
    Group code suggestions by behavioral signature.

    Args:
        codes: List of code suggestion strings
        approved_tests: Tests that have been approved by oracle/user

    Returns:
        {signature_tuple: [list of code indices with this behavior]}
    """
    clusters = defaultdict(list)

    for i, code in enumerate(codes):
        sig = build_behavior_signature(code, approved_tests)
        clusters[sig].append(i)

    return dict(clusters)


def rank_codes_by_clustering(codes: List[str],
                              code_indices: List[int],
                              approved_tests: List[str],
                              execution_matrix: Dict = None) -> List[Tuple[int, str, int, int]]:
    """
    Rank codes using behavioral clustering.

    Primary sort: cluster size (descending) — consensus behavior first
    Secondary sort: number of passing tests (descending) — within cluster

    Args:
        codes: List of code strings
        code_indices: Original indices corresponding to codes
        approved_tests: Oracle/user approved test strings
        execution_matrix: Pre-computed execution results (for passing count)

    Returns:
        List of (original_index, code, cluster_size, passing_tests)
        sorted by (cluster_size DESC, passing_tests DESC)
    """
    if not approved_tests:
        # No approved tests yet — fall back to original order
        return [(code_indices[i], code, 1, 0) for i, code in enumerate(codes)]

    # Build clusters (local indices into codes list)
    clusters = cluster_codes(codes, approved_tests)

    # Build result list
    ranked = []
    for sig, local_indices in clusters.items():
        cluster_size = len(local_indices)

        for local_idx in local_indices:
            code = codes[local_idx]
            orig_idx = code_indices[local_idx]

            # Count passing tests (secondary sort criterion)
            if execution_matrix:
                n_passing = sum(
                    1 for test in approved_tests
                    if execution_matrix.get(test, {}).get(orig_idx) == "pass"
                )
            else:
                # Compute directly if no matrix
                n_passing = sum(
                    1 for test in approved_tests
                    if _passes_test(code, test)
                )

            ranked.append((orig_idx, code, cluster_size, n_passing))

    # Sort: cluster_size DESC, then n_passing DESC
    ranked.sort(key=lambda x: (x[2], x[3]), reverse=True)
    return ranked


def _passes_test(code: str, test: str) -> bool:
    """Quick check if a code passes a single test."""
    input_str, expected = parse_test_assertion(test)
    if input_str is None:
        return False
    success, actual, _ = execute_code(code, input_str)
    return success and actual == expected


def get_cluster_summary(codes: List[str],
                         approved_tests: List[str]) -> dict:
    """
    Get a summary of clustering results for logging/analysis.

    Returns:
        {
            "n_clusters": int,
            "largest_cluster_size": int,
            "largest_cluster_fraction": float,
            "cluster_sizes": list of sizes
        }
    """
    if not approved_tests or not codes:
        return {"n_clusters": len(codes), "largest_cluster_size": 1,
                "largest_cluster_fraction": 1/len(codes) if codes else 0,
                "cluster_sizes": [1] * len(codes)}

    clusters = cluster_codes(codes, approved_tests)
    sizes = sorted([len(v) for v in clusters.values()], reverse=True)

    return {
        "n_clusters": len(clusters),
        "largest_cluster_size": sizes[0] if sizes else 0,
        "largest_cluster_fraction": sizes[0] / len(codes) if codes else 0,
        "cluster_sizes": sizes
    }


if __name__ == "__main__":
    # Quick demonstration
    code1 = "def add(a, b):\n    return a + b"
    code2 = "def add(a, b):\n    return a + b + 0"  # Same behavior
    code3 = "def add(a, b):\n    return a - b"       # Different behavior

    tests = ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]

    ranked = rank_codes_by_clustering([code1, code2, code3], tests)
    print("Ranked codes:")
    for idx, code, cluster_size, n_passing in ranked:
        print(f"  idx={idx} cluster_size={cluster_size} passing={n_passing}")

    summary = get_cluster_summary([code1, code2, code3], tests)
    print(f"\nCluster summary: {summary}")
