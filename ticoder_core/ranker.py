"""
ticoder_core/ranker.py
Implements the discriminative test ranking strategy from Section IV-B-2.

The key formula from the paper:
    s_discr(t) = min(|G+t|, |G-t|) / max(|G+t|, |G-t|)

Where:
- G+t = set of codes that PASS test t
- G-t = set of codes that FAIL test t
- Codes that CRASH are treated as precondition violations (ignored)

Tests with s_discr closest to 1 split codes most evenly
and are therefore shown to the user first.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Tuple
from utils.oracle import execute_code, parse_test_assertion


def execute_test_on_code(code: str, test_str: str) -> str:
    """
    Execute a test assertion on a code suggestion.

    Returns:
        "pass"  - test passes (assertion holds)
        "fail"  - test fails (assertion fails)
        "crash" - code crashes (precondition violation)
    """
    try:
        # Parse the test to get input and expected output
        input_str, expected_output = parse_test_assertion(test_str)

        if input_str is None:
            return "crash"

        # Execute code with this input
        success, actual_output, error = execute_code(code, input_str)

        if not success:
            return "crash"

        # Check if actual output matches expected
        if actual_output == expected_output:
            return "pass"
        else:
            return "fail"

    except Exception:
        return "crash"


def build_execution_matrix(codes: List[str],
                           tests: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Build execution matrix: for each test, record pass/fail/crash for each code.

    Args:
        codes: List of code suggestion strings
        tests: List of test assertion strings
    Returns:
        {test_str: {code_idx: "pass"|"fail"|"crash"}}
    """
    matrix = {}

    for test in tests:
        matrix[test] = {}
        for i, code in enumerate(codes):
            result = execute_test_on_code(code, test)
            matrix[test][i] = result

    return matrix


def compute_s_discr(test: str, code_indices: List[int],
                    execution_matrix: Dict) -> float:
    """
    Compute discriminative score s_discr for a test.
    Formula from Section IV-B-2 of paper.

    Args:
        test: Test assertion string
        code_indices: Original indices of surviving code suggestions
        execution_matrix: Pre-computed execution results
    Returns:
        s_discr score between 0 and 1
    """
    results = execution_matrix.get(test, {})

    # Count codes that pass and fail (ignoring crashes)
    # Use original indices for correct execution matrix lookup
    n_pass = sum(1 for idx in code_indices
                 if results.get(idx) == "pass")
    n_fail = sum(1 for idx in code_indices
                 if results.get(idx) == "fail")

    # s_discr formula
    max_val = max(n_pass, n_fail)
    min_val = min(n_pass, n_fail)

    if max_val == 0:
        return 0.0

    return min_val / max_val


def rank_tests(tests: List[str], code_indices: List[int],
               execution_matrix: Dict) -> List[Tuple[str, float]]:
    """
    Rank tests by discriminative score (highest first).
    This determines the order tests are shown to the user.

    Args:
        tests: List of test strings
        code_indices: Original indices of surviving code suggestions
        execution_matrix: Pre-computed execution results
    Returns:
        List of (test, score) tuples sorted by score descending
    """
    scored = []
    for test in tests:
        score = compute_s_discr(test, code_indices, execution_matrix)
        scored.append((test, score))

    # Sort by score descending (highest discriminative power first)
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def rank_codes(codes: List[str], code_indices: List[int],
               tests: List[str],
               execution_matrix: Dict) -> List[Tuple[int, str, int]]:
    """
    Rank remaining code suggestions by number of tests they pass.
    Final ranking strategy from Section IV-B-3 of paper.

    Each code c gets score = number of tests it passes (d_c).
    Codes ranked in decreasing order of d_c.

    Args:
        codes: List of surviving code strings
        code_indices: Original indices corresponding to codes
        tests: List of test strings
        execution_matrix: Execution results
    Returns:
        List of (original_index, code, score) sorted by score desc
    """
    scored = []
    for code, orig_idx in zip(codes, code_indices):
        score = sum(
            1 for test in tests
            if execution_matrix.get(test, {}).get(orig_idx) == "pass"
        )
        scored.append((orig_idx, code, score))

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored
