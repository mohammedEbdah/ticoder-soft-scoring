"""
ticoder_core/pruner.py
Implements code pruning based on user responses.
Implements Section IV-B-3 of the paper: "Pruning and ranking code suggestions"

Pruning rules:
- PASS -> remove codes that FAIL the test
- FAIL -> remove codes that PASS the test
- FAIL + OUTPUT -> keep only codes that produce the oracle-correct output
- UNDEFINED -> no pruning
"""

import os
import sys
from typing import List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.oracle import execute_code, parse_test_assertion, PASS, FAIL, UNDEFINED


def prune_codes(
    codes: List[str],
    test_str: str,
    user_response: str,
    correct_output=None,
    execution_matrix: dict = None,
    code_indices: Optional[List[int]] = None,
) -> List[int]:
    """
    Prune code suggestions based on user response to a test.

    Args:
        codes: Current list of code suggestions
        test_str: The test assertion shown to user
        user_response: PASS, FAIL, or UNDEFINED
        correct_output: Correct output provided by user (TICODER-OUTPUT only)
        execution_matrix: Pre-computed execution results (optional, for speed)
        code_indices: Original indices for `codes` when using execution_matrix

    Returns:
        List of local indices of SURVIVING (unpruned) codes
    """
    if user_response == UNDEFINED:
        return list(range(len(codes)))

    input_str, expected_output = parse_test_assertion(test_str)
    if input_str is None:
        return list(range(len(codes)))

    surviving: List[int] = []

    for i, code in enumerate(codes):
        original_idx = code_indices[i] if code_indices is not None else i

        # Fast path from execution matrix.
        if execution_matrix and test_str in execution_matrix:
            result = execution_matrix[test_str].get(original_idx, "crash")
        else:
            success, actual_output, _ = execute_code(code, input_str)
            if not success:
                result = "crash"
            elif actual_output == expected_output:
                result = "pass"
            else:
                result = "fail"

        if user_response == PASS:
            # PASS -> remove codes that FAIL the test.
            # Crashes are precondition violations — keep them (paper Section IV-B-3).
            if result in ("pass", "crash"):
                surviving.append(i)

        elif user_response == FAIL:
            if correct_output is not None:
                # OUTPUT variant: keep only codes matching oracle-correct output.
                success, actual_output, _ = execute_code(code, input_str)
                if success and actual_output == correct_output:
                    surviving.append(i)
            else:
                # PASSFAIL variant: remove codes that pass this failing test.
                if result != "pass":
                    surviving.append(i)

    return surviving


def apply_interaction(
    codes: List[str],
    remaining_indices: List[int],
    test_str: str,
    user_response: str,
    correct_output=None,
    execution_matrix: dict = None,
) -> List[int]:
    """
    Apply one round of user interaction to prune codes.
    Returns updated list of surviving original indices.
    """
    remaining_codes = [codes[i] for i in remaining_indices]

    local_surviving = prune_codes(
        remaining_codes,
        test_str,
        user_response,
        correct_output,
        execution_matrix,
        code_indices=remaining_indices,
    )

    new_surviving = [remaining_indices[j] for j in local_surviving]

    # Safety fallback: avoid empty candidate set.
    if len(new_surviving) == 0:
        return remaining_indices

    return new_surviving
