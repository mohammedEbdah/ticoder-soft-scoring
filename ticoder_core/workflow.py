"""
ticoder_core/workflow.py
Full TICODER workflow orchestration.

Phase separation:
- Phase 2 (--phase reproduce): no tools, hard pruning + d_c ranking
- Phase 3 (--phase mcp): configurable Tool 1 and/or Tool 2 ablations
- Phase 4 (--phase noisy): noisy oracle soft-scoring path in noise_recovery.py
"""

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MAX_INTERACTIONS, REPRODUCTION_MODEL, WORKER_COUNT
from mcp_tools.adaptive_interaction import compute_entropy, should_stop
from mcp_tools.soft_scoring import (
    compute_score_delta,
    get_soft_score_summary,
    rank_codes_by_soft_score,
)
from ticoder_core.generator import (
    cache_generations,
    generate_code_suggestions,
    generate_test_suggestions,
    load_from_cache,
)
from ticoder_core.pruner import apply_interaction
from ticoder_core.ranker import build_execution_matrix, rank_codes, rank_tests
from utils.oracle import PASS, evaluate_code_correctness, oracle_response
from utils.safe_io import atomic_write_json


def _rank_survivors(
    code_suggestions: List[str],
    remaining_indices: List[int],
    execution_matrix: Dict,
    used_tests: List[str],
    approved_tests: List[str],
    use_soft_scoring: bool = False,
    scores: Dict[int, int] = None,
) -> List[str]:
    """Rank surviving codes for the current interaction snapshot."""
    del approved_tests  # Approved tests are only needed by Tool 1 entropy.

    surviving_codes = [code_suggestions[i] for i in remaining_indices]

    if use_soft_scoring and scores is not None:
        ranked = rank_codes_by_soft_score(surviving_codes, remaining_indices, scores)
        return [code for _, code, _ in ranked]

    ranked = rank_codes(
        surviving_codes,
        remaining_indices,
        used_tests,
        execution_matrix,
    )
    return [code for _, code, _ in ranked]


def run_ticoder_on_example(
    example: dict,
    variant: str = "passfail",
    use_cache: bool = True,
    use_mcp: bool = False,
    use_tool1: bool = False,
    use_tool2: bool = False,
    model: str = REPRODUCTION_MODEL,
    require_cache: bool = False,
    quiet: bool = False,
) -> dict:
    """
    Run TICODER on a single MBPP example.

    Args:
        example: MBPP example dict with keys: task_id, text, code, test_list.
        variant: "passfail" or "output".
        use_cache: Whether to use cached generations.
        use_mcp: If True, enables Phase 3 tool flags.
        use_tool1: Tool 1, entropy-based early stopping.
        use_tool2: Tool 2, soft scoring with no hard pruning.
        model: Claude model to use.
        require_cache: If True, fail fast when cache for this task is missing.
        quiet: If True, suppress per-task print output.

    Returns:
        Result dict for metrics computation.
    """
    if quiet:

        def print(*_args, **_kwargs):
            pass

    task_id = str(example["task_id"])
    description = example["text"]
    reference_code = example["code"]
    hidden_tests = example["test_list"]

    print(f"\nProcessing task {task_id}: {description[:60]}...")

    func_header = ""
    for line in reference_code.split("\n"):
        if line.strip().startswith("def "):
            func_header = line.strip()
            break

    if not func_header:
        print("  Could not extract function header, skipping.")
        return None

    cached = load_from_cache(task_id) if use_cache else None
    if cached:
        print("  Using cached generations.")
        code_suggestions = cached["code_suggestions"]
        test_suggestions = cached["test_suggestions"]
    else:
        if require_cache:
            raise FileNotFoundError(
                f"Missing cache for task {task_id}. Run --phase reproduce first."
            )

        code_suggestions = generate_code_suggestions(description, func_header, model=model)
        test_suggestions = generate_test_suggestions(description, func_header, model=model)

        if use_cache:
            cache_generations(task_id, code_suggestions, test_suggestions)

    if not code_suggestions or not test_suggestions:
        print("  Generation failed, skipping.")
        return None

    print(
        f"  Building execution matrix ({len(code_suggestions)} codes x "
        f"{len(test_suggestions)} tests)..."
    )
    execution_matrix = build_execution_matrix(code_suggestions, test_suggestions)

    n_correct = sum(
        1 for code in code_suggestions if evaluate_code_correctness(code, hidden_tests)
    )
    print(f"  Correct suggestions: {n_correct}/{len(code_suggestions)}")

    active_tool1 = bool(use_mcp and use_tool1)
    use_soft_scoring = bool(use_mcp and use_tool2)

    remaining_indices = list(range(len(code_suggestions)))
    remaining_tests = list(test_suggestions)
    used_tests: List[str] = []
    approved_tests: List[str] = []
    interactions_log: List[dict] = []
    ranked_codes_per_m: Dict[int, List[str]] = {}
    adaptive_m_used = MAX_INTERACTIONS
    scores = {i: 0 for i in range(len(code_suggestions))} if use_soft_scoring else None

    for m in range(1, MAX_INTERACTIONS + 1):
        if active_tool1:
            surviving_codes_now = [code_suggestions[i] for i in remaining_indices]
            stop_early, stop_reason = should_stop(
                surviving_codes_now,
                approved_tests,
                remaining_tests,
                current_m=m,
                original_n_codes=len(code_suggestions),
            )
            if stop_early:
                adaptive_m_used = max(1, m - 1)
                print(
                    f"  [Tool 1] Early stopping at m={m}: {stop_reason} "
                    f"(used {adaptive_m_used} interactions)"
                )
                snapshot = _rank_survivors(
                    code_suggestions,
                    remaining_indices,
                    execution_matrix,
                    used_tests,
                    approved_tests,
                    use_soft_scoring=use_soft_scoring,
                    scores=scores,
                )
                for remaining_m in range(m, MAX_INTERACTIONS + 1):
                    ranked_codes_per_m[remaining_m] = snapshot
                break

        if use_soft_scoring and not remaining_tests and used_tests:
            remaining_tests = list(used_tests)

        if not remaining_tests:
            snapshot = _rank_survivors(
                code_suggestions,
                remaining_indices,
                execution_matrix,
                used_tests,
                approved_tests,
                use_soft_scoring=use_soft_scoring,
                scores=scores,
            )
            for remaining_m in range(m, MAX_INTERACTIONS + 1):
                ranked_codes_per_m[remaining_m] = snapshot
            break

        ranked_tests = rank_tests(remaining_tests, remaining_indices, execution_matrix)
        if not ranked_tests:
            snapshot = _rank_survivors(
                code_suggestions,
                remaining_indices,
                execution_matrix,
                used_tests,
                approved_tests,
                use_soft_scoring=use_soft_scoring,
                scores=scores,
            )
            for remaining_m in range(m, MAX_INTERACTIONS + 1):
                ranked_codes_per_m[remaining_m] = snapshot
            break

        top_test, top_score = ranked_tests[0]
        remaining_tests.remove(top_test)
        used_tests.append(top_test)

        oracle_result = oracle_response(reference_code, top_test, variant)
        response = oracle_result["response"]

        if response == PASS and top_test not in approved_tests:
            approved_tests.append(top_test)

        print(f"  m={m}: test='{top_test[:50]}' response={response}")

        summary = None
        if use_soft_scoring:
            for idx in remaining_indices:
                scores[idx] += compute_score_delta(
                    idx,
                    top_test,
                    response,
                    execution_matrix,
                )

            ranked = rank_codes_by_soft_score(
                code_suggestions,
                remaining_indices,
                scores,
            )
            ranked_codes_per_m[m] = [code for _, code, _ in ranked]

            summary = get_soft_score_summary(scores)
            print(
                f"  [Tool 2] Soft Score: top={summary['top_score']}, "
                f"mean={summary['mean_score']:.2f}"
            )
        else:
            remaining_indices = apply_interaction(
                code_suggestions,
                remaining_indices,
                top_test,
                response,
                oracle_result.get("correct_output"),
                execution_matrix,
            )
            ranked = rank_codes(
                [code_suggestions[i] for i in remaining_indices],
                remaining_indices,
                used_tests,
                execution_matrix,
            )
            ranked_codes_per_m[m] = [code for _, code, _ in ranked]

        log_entry = {
            "m": m,
            "test": top_test,
            "score": top_score,
            "response": response,
            "n_surviving": len(remaining_indices),
            "n_approved_tests": len(approved_tests),
        }

        if active_tool1:
            log_entry["entropy"] = compute_entropy(
                [code_suggestions[i] for i in remaining_indices],
                approved_tests,
            )
        if use_soft_scoring and summary is not None:
            log_entry["top_soft_score"] = summary["top_score"]

        interactions_log.append(log_entry)

    if ranked_codes_per_m:
        latest_ranked = ranked_codes_per_m[max(ranked_codes_per_m.keys())]
    else:
        latest_ranked = _rank_survivors(
            code_suggestions,
            remaining_indices,
            execution_matrix,
            used_tests,
            approved_tests,
            use_soft_scoring=use_soft_scoring,
            scores=scores,
        )

    for m in range(1, MAX_INTERACTIONS + 1):
        if m not in ranked_codes_per_m:
            ranked_codes_per_m[m] = latest_ranked

    print(f"  Final surviving codes: {len(remaining_indices)}")

    if active_tool1 and len(interactions_log) == MAX_INTERACTIONS:
        adaptive_m_used = MAX_INTERACTIONS

    return {
        "task_id": task_id,
        "n_total": len(code_suggestions),
        "n_correct": n_correct,
        "ranked_codes_per_m": ranked_codes_per_m,
        "hidden_tests": hidden_tests,
        "interactions": interactions_log,
        "variant": variant,
        "adaptive_m_used": adaptive_m_used if use_mcp else MAX_INTERACTIONS,
    }


def run_ticoder_on_dataset(
    examples: List[dict],
    variant: str = "passfail",
    use_cache: bool = True,
    use_mcp: bool = False,
    use_tool1: bool = False,
    use_tool2: bool = False,
    model: str = REPRODUCTION_MODEL,
    results_file: str = None,
    require_cache: bool = False,
) -> List[dict]:
    """
    Run TICODER on a dataset with parallel processing and checkpointing.

    Uses ThreadPoolExecutor with os.cpu_count()-1 workers to process multiple
    tasks concurrently. Each worker spawns subprocesses for code execution
    timeouts via oracle.py.
    """
    results: List[dict] = []
    checkpoint_file = results_file or f"results/ticoder_{variant}_checkpoint.json"
    failures_file = f"{checkpoint_file}.failures.json"
    failures: List[dict] = []

    existing_ids = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            results = json.load(f)
            existing_ids = {r["task_id"] for r in results}
        print(f"  Resuming from checkpoint: {len(results)} examples done.")

    tasks_to_run = [ex for ex in examples if str(ex["task_id"]) not in existing_ids]

    if not tasks_to_run:
        print("  All tasks already completed.")
    else:
        n_workers = WORKER_COUNT
        total = len(tasks_to_run)
        print(f"  Processing {total} tasks with {n_workers} parallel workers...")

        save_lock = threading.Lock()
        counter = [0]

        def _run_one(example):
            return run_ticoder_on_example(
                example,
                variant=variant,
                use_cache=use_cache,
                use_mcp=use_mcp,
                use_tool1=use_tool1,
                use_tool2=use_tool2,
                model=model,
                require_cache=require_cache,
                quiet=True,
            )

        os.makedirs("results", exist_ok=True)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_one, ex): ex for ex in tasks_to_run}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    ex = futures[future]
                    failure = {
                        "task_id": str(ex.get("task_id", "?")),
                        "error_type": type(e).__name__,
                        "error": str(e),
                    }
                    failures.append(failure)
                    atomic_write_json(failures_file, failures, indent=2)
                    print(
                        f"  Task {failure['task_id']} failed: "
                        f"{failure['error_type']}: {failure['error']}"
                    )
                    continue

                if result:
                    with save_lock:
                        results.append(result)
                        counter[0] += 1
                        atomic_write_json(checkpoint_file, results, indent=2)
                        if counter[0] % 25 == 0 or counter[0] == total:
                            print(
                                f"  TICODER-{variant.upper()}: "
                                f"{counter[0]}/{total} tasks done"
                            )

    os.makedirs("results", exist_ok=True)
    atomic_write_json(checkpoint_file, results, indent=2)

    if failures:
        raise RuntimeError(
            f"{len(failures)} task(s) failed for {variant}; "
            f"partial results saved to {checkpoint_file}, "
            f"failures saved to {failures_file}"
        )
    if os.path.exists(failures_file):
        os.remove(failures_file)

    print(
        f"\n  Completed {variant.upper()}! {len(results)} results "
        f"saved to {checkpoint_file}"
    )
    return results
