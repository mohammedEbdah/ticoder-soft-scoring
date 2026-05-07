"""
mcp_tools/noise_recovery.py
Noisy oracle simulation for Phase 4.

This phase injects controlled PASS/FAIL flips and evaluates the soft-scoring
ranking path. UNDEFINED responses are never flipped.
"""

import os
import random
import sys
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.oracle import FAIL, PASS, oracle_response

NOISE_LEVELS = [1]  # Single random flip out of m=5 interactions (20% noise)


def noisy_oracle_response(
    reference_code: str,
    test_str: str,
    variant: str = "passfail",
    flip: bool = False,
) -> dict:
    """
    Oracle response with deterministic noise injection.

    If `flip` is True, flips PASS <-> FAIL for this interaction.
    UNDEFINED is never flipped.
    """
    true_result = oracle_response(reference_code, test_str, variant)
    true_response = true_result["response"]

    result = dict(true_result)
    result["original_response"] = true_response
    result["noisy"] = False

    if flip and true_response in (PASS, FAIL):
        result["response"] = FAIL if true_response == PASS else PASS
        if result["response"] == FAIL:
            # For flipped PASS->FAIL, we cannot trust a correct output.
            result["correct_output"] = None
        result["noisy"] = True

    return result


def run_noise_experiment_on_example(
    example: dict,
    n_flips: int,
    variant: str = "passfail",
    use_soft_scoring: bool = True,
) -> dict:
    """
    Run one noisy-oracle TICODER pass.

    Random flipping: randomly selects n_flips interactions (out of m=5) to flip,
    seeded by task_id for reproducibility.
    When use_soft_scoring=True, uses soft scoring (+1/-1/0) — codes survive noise.
    When use_soft_scoring=False, uses hard pruning — a single bad flip can kill codes.
    Includes test recycling when all unique tests have been used (soft scoring only).
    """
    from config import MAX_INTERACTIONS
    from mcp_tools.soft_scoring import (
        compute_score_delta,
        get_soft_score_summary,
        rank_codes_by_soft_score,
    )
    from ticoder_core.generator import load_from_cache
    from ticoder_core.pruner import apply_interaction
    from ticoder_core.ranker import build_execution_matrix, rank_codes, rank_tests
    from utils.oracle import PASS, evaluate_code_correctness

    task_id = str(example["task_id"])
    reference_code = example["code"]
    hidden_tests = example["test_list"]

    cached = load_from_cache(task_id)
    if not cached:
        return None

    code_suggestions = cached.get("code_suggestions", [])
    test_suggestions = cached.get("test_suggestions", [])
    if not code_suggestions or not test_suggestions:
        return None

    execution_matrix = build_execution_matrix(code_suggestions, test_suggestions)

    n_correct = sum(
        1 for code in code_suggestions if evaluate_code_correctness(code, hidden_tests)
    )

    # Pick which interaction(s) to flip randomly, seeded by task_id for reproducibility
    rng = random.Random(int(task_id))
    flip_set = set(rng.sample(range(1, MAX_INTERACTIONS + 1), n_flips))
    remaining_indices = list(range(len(code_suggestions)))
    remaining_tests = list(test_suggestions)
    used_tests: List[str] = []
    approved_tests: List[str] = []
    scores = {i: 0 for i in remaining_indices} if use_soft_scoring else None
    ranked_codes_per_m: Dict[int, List[str]] = {}
    interactions_log: List[dict] = []
    noise_flips = 0

    for m in range(1, MAX_INTERACTIONS + 1):
        # Soft scoring recycles tests; hard pruning does not.
        if use_soft_scoring and not remaining_tests and used_tests:
            remaining_tests = list(used_tests)

        if not remaining_tests:
            if use_soft_scoring:
                ranked = rank_codes_by_soft_score(
                    code_suggestions, remaining_indices, scores,
                )
            else:
                ranked = rank_codes(
                    [code_suggestions[i] for i in remaining_indices],
                    remaining_indices, used_tests, execution_matrix,
                )
            snapshot = [code for _, code, _ in ranked]
            for rm in range(m, MAX_INTERACTIONS + 1):
                ranked_codes_per_m[rm] = snapshot
            break

        ranked_tests = rank_tests(remaining_tests, remaining_indices, execution_matrix)
        if not ranked_tests:
            if use_soft_scoring:
                ranked = rank_codes_by_soft_score(
                    code_suggestions, remaining_indices, scores,
                )
            else:
                ranked = rank_codes(
                    [code_suggestions[i] for i in remaining_indices],
                    remaining_indices, used_tests, execution_matrix,
                )
            snapshot = [code for _, code, _ in ranked]
            for rm in range(m, MAX_INTERACTIONS + 1):
                ranked_codes_per_m[rm] = snapshot
            break

        top_test, top_score = ranked_tests[0]
        remaining_tests.remove(top_test)
        used_tests.append(top_test)

        oracle_result = noisy_oracle_response(
            reference_code,
            top_test,
            variant=variant,
            flip=(m in flip_set),
        )
        response = oracle_result["response"]
        was_noisy = bool(oracle_result.get("noisy", False))

        if was_noisy:
            noise_flips += 1
        if response == PASS and top_test not in approved_tests:
            approved_tests.append(top_test)

        if use_soft_scoring:
            for idx in remaining_indices:
                scores[idx] += compute_score_delta(
                    idx, top_test, response, execution_matrix,
                )
            ranked = rank_codes_by_soft_score(
                code_suggestions, remaining_indices, scores,
            )
            ranked_codes_per_m[m] = [code for _, code, _ in ranked]
        else:
            remaining_indices = apply_interaction(
                code_suggestions, remaining_indices, top_test,
                response, oracle_result.get("correct_output"), execution_matrix,
            )
            ranked = rank_codes(
                [code_suggestions[i] for i in remaining_indices],
                remaining_indices, used_tests, execution_matrix,
            )
            ranked_codes_per_m[m] = [code for _, code, _ in ranked]

        log_entry = {
            "m": m,
            "test": top_test,
            "score": top_score,
            "response": response,
            "original_response": oracle_result.get("original_response"),
            "was_noisy": was_noisy,
            "n_surviving": len(remaining_indices),
            "n_approved_tests": len(approved_tests),
        }
        if use_soft_scoring:
            summary = get_soft_score_summary(scores)
            log_entry["top_soft_score"] = summary["top_score"]
        interactions_log.append(log_entry)

    if ranked_codes_per_m:
        latest = ranked_codes_per_m[max(ranked_codes_per_m.keys())]
    else:
        if use_soft_scoring:
            ranked = rank_codes_by_soft_score(
                code_suggestions, remaining_indices, scores,
            )
        else:
            ranked = rank_codes(
                [code_suggestions[i] for i in remaining_indices],
                remaining_indices, used_tests, execution_matrix,
            )
        latest = [code for _, code, _ in ranked]

    for rm in range(1, MAX_INTERACTIONS + 1):
        ranked_codes_per_m.setdefault(rm, latest)

    return {
        "task_id": task_id,
        "n_total": len(code_suggestions),
        "n_correct": n_correct,
        "ranked_codes_per_m": ranked_codes_per_m,
        "hidden_tests": hidden_tests,
        "interactions": interactions_log,
        "variant": variant,
        "adaptive_m_used": MAX_INTERACTIONS,
        "n_flips": n_flips,
        "noise_flips": noise_flips,
        "use_soft_scoring": use_soft_scoring,
    }


def print_noise_robustness_table(
    results_by_noise_level: Dict[int, List[dict]],
    phase2_baseline_m5: float = None,
):
    """Print pass@1@5 robustness table across noisy runs."""
    from utils.metrics import compute_pass_at_k_at_m

    print("\nNOISE ROBUSTNESS TABLE (pass@1@5)")
    print("==================================")

    if phase2_baseline_m5 is not None:
        print(f"  0/5 flips (Phase 2 reference): {phase2_baseline_m5:.2f}%")

    for n_flips in sorted(results_by_noise_level):
        results = results_by_noise_level[n_flips]
        score = compute_pass_at_k_at_m(results, k=1).get(5, 0.0)
        if phase2_baseline_m5 is None:
            delta = 0.0
        else:
            delta = score - phase2_baseline_m5
        print(f"  {n_flips}/5 flips: {score:.2f}%  (delta: {delta:+.2f}%)")
