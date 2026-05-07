"""
utils/metrics.py
Computes pass@k and pass@k@m metrics as defined in the paper (Section VII-B).

pass@k: Standard metric - probability that at least one of k random
        samples is correct. Used for baseline.

pass@k@m: TICODER metric - after m user interactions, checks if any
          of the TOP k RANKED suggestions is correct. This is
          deterministic (not statistical) since TICODER outputs a ranked list.
"""

import numpy as np
from typing import List


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute pass@k given:
    - n: total number of generated samples
    - c: number of correct samples
    - k: number of samples to check

    Uses the unbiased estimator from the Codex paper (Chen et al. 2021):
    pass@k = 1 - C(n-c, k) / C(n, k)

    This avoids numerical issues with large combinations.
    """
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(
        1.0 - k / np.arange(n - c + 1, n + 1)
    )


def compute_baseline_pass_at_k(results: List[dict], k: int = 1) -> float:
    """
    Compute baseline pass@k across all examples.

    Args:
        results: List of dicts with keys:
                 - 'n_total': number of generated suggestions
                 - 'n_correct': number of correct suggestions
        k: k value for pass@k

    Returns:
        Mean pass@k across all examples (as percentage)
    """
    scores = []
    for r in results:
        score = pass_at_k(r["n_total"], r["n_correct"], k)
        scores.append(score)
    return np.mean(scores) * 100


def compute_pass_at_k_at_m(results: List[dict], k: int = 1) -> dict:
    """
    Compute pass@k@m for m = 1 to 5 (same as Table IV in paper).

    Pre-computes correctness for all unique (code, hidden_tests) pairs
    in parallel, then looks up results — avoids sequential subprocess spawns.

    Args:
        results: List of dicts with keys:
                 - 'ranked_codes_per_m': dict {m: [code1, code2, ...]}
                   ranked codes after m interactions
                 - 'hidden_tests': list of test strings

    Returns:
        Dict {m: pass@k@m_percentage} for m in 1..5
    """
    import os
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from utils.oracle import evaluate_code_correctness

    # Step 1: Collect all unique (code, tests_key) pairs we need to evaluate.
    # Use frozenset of hidden_tests as the key since test lists are the same
    # per-task but differ across tasks.
    pairs_to_eval = {}  # (code, tests_tuple) -> None (dedup)
    for result in results:
        tests_tuple = tuple(result["hidden_tests"])
        for m in range(1, 6):
            rcm = result.get("ranked_codes_per_m", {})
            ranked_codes = rcm.get(m) or rcm.get(str(m)) or []
            for code in ranked_codes[:k]:
                pairs_to_eval[(code, tests_tuple)] = None

    # Step 2: Evaluate all unique pairs in parallel.
    correctness_cache = {}  # (code, tests_tuple) -> bool
    pairs_list = list(pairs_to_eval.keys())

    if pairs_list:
        n_workers = max(1, (os.cpu_count() or 2) - 1)
        lock = threading.Lock()
        done = [0]
        total = len(pairs_list)

        def _eval(pair):
            code, tests_tuple = pair
            return pair, evaluate_code_correctness(code, list(tests_tuple))

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_eval, p): p for p in pairs_list}
            for future in as_completed(futures):
                try:
                    pair, result_val = future.result()
                    with lock:
                        correctness_cache[pair] = result_val
                        done[0] += 1
                        if done[0] % 200 == 0 or done[0] == total:
                            print(f"  Metrics eval: {done[0]}/{total} unique codes checked")
                except Exception:
                    pair = futures[future]
                    with lock:
                        correctness_cache[pair] = False
                        done[0] += 1

    # Step 3: Look up results from cache.
    m_scores = {m: [] for m in range(1, 6)}

    for result in results:
        tests_tuple = tuple(result["hidden_tests"])

        for m in range(1, 6):
            rcm = result.get("ranked_codes_per_m", {})
            ranked_codes = rcm.get(m) or rcm.get(str(m)) or []
            top_k = ranked_codes[:k]
            correct = any(
                correctness_cache.get((code, tests_tuple), False)
                for code in top_k
            )
            m_scores[m].append(1.0 if correct else 0.0)

    return {
        m: np.mean(scores) * 100
        for m, scores in m_scores.items()
    }


def print_results_table(baseline_pass1: float, baseline_pass30: float,
                        passfail_results: dict, output_results: dict,
                        model_name: str, dataset_name: str):
    """
    Print results in the same format as Table IV in the paper.
    """
    print(f"\n{'='*80}")
    print(f"Results for {model_name} on {dataset_name}")
    print(f"{'='*80}")
    print(f"{'Metric':<30} {'Value':>10}")
    print(f"{'-'*40}")
    print(f"{'Baseline pass@1':<30} {baseline_pass1:>9.2f}%")
    print(f"{'Baseline pass@30':<30} {baseline_pass30:>9.2f}%")
    print()
    print(f"{'TICODER-PASSFAIL':}")
    for m in range(1, 6):
        val = passfail_results.get(m, 0)
        print(f"  {'pass@1@' + str(m):<28} {val:>9.2f}%")
    print()
    print(f"{'TICODER-OUTPUT':}")
    for m in range(1, 6):
        val = output_results.get(m, 0)
        print(f"  {'pass@1@' + str(m):<28} {val:>9.2f}%")
    print(f"{'='*80}\n")


def compute_absolute_improvement(baseline: float, ticoder_m5: float) -> float:
    """Compute absolute improvement like the paper reports."""
    return ticoder_m5 - baseline


# ══════════════════════════════════════════════════════════════════════════════
# MCP Tool Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_adaptive_stopping_stats(results: List[dict]) -> dict:
    """
    Compute statistics for adaptive stopping (MCP Tool 1: adaptive_interaction).

    Measures (from CONTEXT.md):
    - avg_m_to_converge: mean m where Tool 1 stops early
    - pct_stopped_early: % of examples where H < 0.5 before m=5
    - Efficiency gain percentage
    - Distribution of stopping points

    Args:
        results: List of result dicts with 'adaptive_m_used' key

    Returns:
        Dict with adaptive stopping statistics
    """
    m_values = [r.get("adaptive_m_used", 5) for r in results]

    if not m_values:
        return {"avg_m": 5.0, "efficiency_gain_pct": 0.0, "pct_stopped_early": 0.0}

    avg_m = np.mean(m_values)
    efficiency_gain = (1 - avg_m / 5.0) * 100  # % fewer interactions than fixed m=5

    # pct_stopped_early: % of examples where Tool 1 stopped before m=5
    n_stopped_early = sum(1 for m in m_values if m < 5)
    pct_stopped_early = (n_stopped_early / len(m_values)) * 100

    # Distribution of stopping points
    from collections import Counter
    distribution = dict(Counter(m_values))

    return {
        "avg_m": round(float(avg_m), 2),
        "median_m": round(float(np.median(m_values)), 1),
        "min_m": int(min(m_values)),
        "max_m": int(max(m_values)),
        "efficiency_gain_pct": round(float(efficiency_gain), 1),
        "pct_stopped_early": round(float(pct_stopped_early), 1),
        "distribution": distribution
    }


def print_mcp_results_table(baseline_pass1: float, baseline_pass30: float,
                             phase2_passfail: dict, phase2_output: dict,
                             phase3_passfail: dict, phase3_output: dict,
                             adaptive_stats: dict,
                             model_name: str, dataset_name: str):
    """
    Print comprehensive results comparing Phase 2 (reproduction) vs Phase 3 (MCP).

    Shows:
    - Baseline metrics
    - Phase 2 pass@1@m (paper reproduction)
    - Phase 3 pass@1@m (with MCP tools)
    - Delta (improvement from MCP tools)
    - Adaptive stopping efficiency
    """
    print(f"\n{'='*90}")
    print(f"COMPREHENSIVE RESULTS: {model_name} on {dataset_name}")
    print(f"{'='*90}")

    # Baseline
    print(f"\n  Baseline pass@1:  {baseline_pass1:>8.2f}%")
    print(f"  Baseline pass@30: {baseline_pass30:>8.2f}%")

    # Phase 2 vs Phase 3 comparison
    print(f"\n  {'m':<4} {'Phase 2 PF':>12} {'Phase 3 PF':>12} {'Delta PF':>10}"
          f" | {'Phase 2 OUT':>12} {'Phase 3 OUT':>12} {'Delta OUT':>10}")
    print(f"  {'-'*76}")

    for m in range(1, 6):
        p2_pf = phase2_passfail.get(m, 0)
        p3_pf = phase3_passfail.get(m, 0)
        d_pf = p3_pf - p2_pf

        p2_out = phase2_output.get(m, 0)
        p3_out = phase3_output.get(m, 0)
        d_out = p3_out - p2_out

        print(f"  {m:<4} {p2_pf:>11.2f}% {p3_pf:>11.2f}% {d_pf:>+9.2f}%"
              f" | {p2_out:>11.2f}% {p3_out:>11.2f}% {d_out:>+9.2f}%")

    # Absolute improvement over baseline
    p2_pf5 = phase2_passfail.get(5, 0)
    p3_pf5 = phase3_passfail.get(5, 0)
    p2_out5 = phase2_output.get(5, 0)
    p3_out5 = phase3_output.get(5, 0)

    print(f"\n  Absolute improvement over baseline (pass@1@5 - baseline pass@1):")
    print(f"    Phase 2 PASSFAIL: {p2_pf5 - baseline_pass1:>+.2f}%")
    print(f"    Phase 3 PASSFAIL: {p3_pf5 - baseline_pass1:>+.2f}%")
    print(f"    Phase 2 OUTPUT:   {p2_out5 - baseline_pass1:>+.2f}%")
    print(f"    Phase 3 OUTPUT:   {p3_out5 - baseline_pass1:>+.2f}%")

    # Adaptive stopping stats (RQ2)
    if adaptive_stats:
        print(f"\n  {'-'*50}")
        print(f"  ADAPTIVE STOPPING (MCP Tool 1)")
        print(f"  {'-'*50}")
        print(f"  avg_m_to_converge:          {adaptive_stats['avg_m']:.2f} / 5.0")
        print(f"  Median interactions used:   {adaptive_stats['median_m']:.1f}")
        print(f"  pct_stopped_early:          {adaptive_stats.get('pct_stopped_early', 0):.1f}%"
              f" of examples stopped before m=5")
        print(f"  Efficiency gain:            {adaptive_stats['efficiency_gain_pct']:.1f}%"
              f" fewer interactions")
        print(f"  Range:                      [{adaptive_stats['min_m']}, "
              f"{adaptive_stats['max_m']}]")
        if "distribution" in adaptive_stats:
            print(f"  Distribution of stopping m: {adaptive_stats['distribution']}")

    print(f"{'='*90}\n")


def print_noise_results(noise_results: dict, phase2_baseline_m5: float = None):
    """
    Print noise robustness results, one row per noise level.
    """
    from mcp_tools.noise_recovery import print_noise_robustness_table
    print_noise_robustness_table(noise_results, phase2_baseline_m5=phase2_baseline_m5)
