"""
mcp_tools/soft_scoring.py
Tool 2: Soft scoring, replacing hard pruning + confidence ranking.

Instead of permanently removing codes after one disagreeing test, this module
accumulates agreement scores across interactions:
- +1 for agreement with the oracle
- -1 for disagreement
-  0 for crashes, precondition violations, or UNDEFINED oracle responses
"""

import os
import sys
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.oracle import FAIL, PASS, UNDEFINED


def compute_score_delta(
    code_idx: int,
    test_str: str,
    oracle_resp: str,
    execution_matrix: dict,
) -> int:
    """
    Compute score delta for one code on one test given oracle response.

    Args:
        code_idx: Original index of the code in code_suggestions.
        test_str: The test assertion string.
        oracle_resp: Oracle response: PASS, FAIL, or UNDEFINED.
        execution_matrix: {test_str: {code_idx: "pass"|"fail"|"crash"}}.

    Returns:
        +1 for agreement, -1 for disagreement, or 0 for uncertainty.
    """
    if oracle_resp == UNDEFINED:
        return 0

    execution_result = execution_matrix.get(test_str, {}).get(code_idx, "crash")

    if oracle_resp == PASS:
        if execution_result == "pass":
            return 1
        if execution_result == "fail":
            return -1
        return 0

    if oracle_resp == FAIL:
        if execution_result == "fail":
            return 1
        if execution_result == "pass":
            return -1
        return 0

    return 0


def rank_codes_by_soft_score(
    codes: List[str],
    code_indices: List[int],
    scores: Dict[int, int],
) -> List[Tuple[int, str, int]]:
    """
    Rank codes by cumulative soft score.

    Primary sort: score descending. Tiebreaker: original index ascending.

    Args:
        codes: Code strings corresponding to code_indices.
        code_indices: Original indices for the given code strings.
        scores: {original_code_index: cumulative_score}.

    Returns:
        List of (original_index, code_string, score) sorted best-first.
    """
    ranked = [
        (orig_idx, code, scores.get(orig_idx, 0))
        for code, orig_idx in zip(codes, code_indices)
    ]
    ranked.sort(key=lambda item: (-item[2], item[0]))
    return ranked


def get_soft_score_summary(scores: Dict[int, int]) -> dict:
    """
    Return summary statistics for logging.

    Returns:
        {
            "n_codes": int,
            "top_score": int,
            "mean_score": float,
            "min_score": int,
            "score_spread": int,
        }
    """
    if not scores:
        return {
            "n_codes": 0,
            "top_score": 0,
            "mean_score": 0.0,
            "min_score": 0,
            "score_spread": 0,
        }

    values = list(scores.values())
    top_score = max(values)
    min_score = min(values)

    return {
        "n_codes": len(values),
        "top_score": top_score,
        "mean_score": sum(values) / len(values),
        "min_score": min_score,
        "score_spread": top_score - min_score,
    }
