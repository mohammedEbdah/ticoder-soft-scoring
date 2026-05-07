"""
experiments/run_all.py
Main entry point for TICODER experiment phases.
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    FULL_SIZE,
    NUM_CODE_SUGGESTIONS,
    NUM_TEST_SUGGESTIONS,
    PILOT_SIZE,
    REPRODUCTION_MODEL,
    TEMPERATURE,
)
from data.download_datasets import download_mbpp, load_mbpp
from ticoder_core.workflow import run_ticoder_on_dataset
from utils.metrics import (
    compute_adaptive_stopping_stats,
    compute_baseline_pass_at_k,
    compute_pass_at_k_at_m,
    print_results_table,
)
from utils.oracle import evaluate_code_correctness
from utils.safe_io import atomic_write_json


TOOL_CONFIGS = {
    "tool1": {
        "phase": "Phase 3a",
        "label": "Tool 1",
        "description": "entropy + hard",
        "use_tool1": True,
        "use_tool2": False,
    },
    "tool2": {
        "phase": "Phase 3b",
        "label": "Tool 2",
        "description": "soft scoring",
        "use_tool1": False,
        "use_tool2": True,
    },
    "both": {
        "phase": "Phase 3c",
        "label": "Both",
        "description": "entropy + soft",
        "use_tool1": True,
        "use_tool2": True,
    },
}


def ensure_phase_cache_exists(examples: List[dict], cache_dir: str = "cache"):
    """Fail fast if any task cache is missing."""
    missing = []
    for ex in examples:
        task_id = str(ex["task_id"])
        cache_path = os.path.join(cache_dir, f"{task_id}.json")
        if not os.path.exists(cache_path):
            missing.append(task_id)

    if missing:
        preview = ", ".join(missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise FileNotFoundError(
            "Phase requires cached generations from Phase 2, but cache is missing for "
            f"{len(missing)} task(s): {preview}{suffix}. "
            "Run --phase reproduce first."
        )


def run_baseline(examples: list, model: str, require_cache: bool = False) -> dict:
    """
    Compute baseline pass@1 and pass@N without interactive TICODER steps.

    Saves results incrementally to results/baseline_results.json after every task.
    If require_cache=True, baseline will only read cached generations.
    """
    from ticoder_core.generator import (
        cache_generations,
        generate_code_suggestions,
        generate_test_suggestions,
        load_from_cache,
    )

    print("\n" + "=" * 60)
    print("Running BASELINE (no TICODER)")
    print("=" * 60)

    baseline_checkpoint = "results/baseline_results.json"
    baseline_results = []
    existing_ids = set()

    os.makedirs("results", exist_ok=True)
    if os.path.exists(baseline_checkpoint):
        with open(baseline_checkpoint, "r", encoding="utf-8") as f:
            baseline_results = json.load(f)
            existing_ids = {r["task_id"] for r in baseline_results}
        print(f"  Resuming baseline from checkpoint: {len(baseline_results)} tasks done.")

    def _eval_one(example):
        task_id = str(example["task_id"])
        cached = load_from_cache(task_id)
        if cached and cached.get("code_suggestions"):
            codes = cached["code_suggestions"]
            if not cached.get("test_suggestions"):
                if require_cache:
                    return None
                func_header = ""
                for line in example["code"].split("\n"):
                    if line.strip().startswith("def "):
                        func_header = line.strip()
                        break
                tests = generate_test_suggestions(example["text"], func_header, model=model)
                cache_generations(task_id, codes, tests)
        else:
            if require_cache:
                return None
            func_header = ""
            for line in example["code"].split("\n"):
                if line.strip().startswith("def "):
                    func_header = line.strip()
                    break
            codes = generate_code_suggestions(example["text"], func_header, model=model)
            tests = generate_test_suggestions(example["text"], func_header, model=model)
            cache_generations(task_id, codes, tests)

        hidden_tests = example["test_list"]
        n_correct = sum(1 for code in codes if evaluate_code_correctness(code, hidden_tests))
        return {"task_id": task_id, "n_total": len(codes), "n_correct": n_correct}

    tasks_to_eval = [ex for ex in examples if str(ex["task_id"]) not in existing_ids]

    if not tasks_to_eval:
        print("  All baseline tasks already evaluated.")
    else:
        n_workers = max(1, (os.cpu_count() or 2) - 1)
        total = len(tasks_to_eval)
        print(f"  Evaluating {total} tasks with {n_workers} parallel workers...")

        save_lock = threading.Lock()
        counter = [0]

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_eval_one, ex): ex for ex in tasks_to_eval}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    print(f"  Baseline task failed: {e}")
                    continue

                if result:
                    with save_lock:
                        baseline_results.append(result)
                        counter[0] += 1
                        atomic_write_json(baseline_checkpoint, baseline_results, indent=2)
                        if counter[0] % 25 == 0 or counter[0] == total:
                            print(f"  Baseline: {counter[0]}/{total} tasks done")

    pass1 = compute_baseline_pass_at_k(baseline_results, k=1)
    pass_n = compute_baseline_pass_at_k(baseline_results, k=NUM_CODE_SUGGESTIONS)

    print(f"\nBaseline pass@1: {pass1:.2f}%")
    print(f"Baseline pass@{NUM_CODE_SUGGESTIONS}: {pass_n:.2f}%")

    return {"pass1": pass1, "pass_at_n": pass_n, "results": baseline_results}


def _load_phase2_scores(variant: str) -> dict:
    """Load Phase 2 pass@1@m scores for a variant, if available."""
    phase2_file = "results/final_reproduce.json"
    if not os.path.exists(phase2_file):
        return {}

    with open(phase2_file, "r", encoding="utf-8") as f:
        phase2_data = json.load(f)

    scores = phase2_data.get(f"ticoder_{variant}", {}).get("m_scores", {})
    return {int(k): v for k, v in scores.items()}


def _format_score(score, phase2_score=None):
    if score is None:
        return "N/A".rjust(18)
    if phase2_score is None:
        return f"{score:>17.2f}%"
    return f"{score:>8.2f}% ({score - phase2_score:+6.2f})"


def _print_phase3_ablation_table(variant: str, config_scores: dict):
    """Print Phase 2 vs Phase 3 ablation comparison for one variant."""
    phase2_scores = _load_phase2_scores(variant)

    print("\n" + "=" * 100)
    print(f"Phase 3 Ablation Comparison ({variant.upper()})")
    print("=" * 100)
    print(
        f"{'Metric':<12} | {'Phase 2 (paper)':>17} | "
        f"{'Phase 3a (Tool 1)':>18} | {'Phase 3b (Tool 2)':>18} | "
        f"{'Phase 3c (Both)':>18}"
    )
    print(
        f"{'Method':<12} | {'hard pruning + d_c':>17} | "
        f"{'entropy + hard':>18} | {'soft scoring':>18} | "
        f"{'entropy + soft':>18}"
    )
    print("-" * 100)

    for m in range(1, 6):
        p2 = phase2_scores.get(m)
        t1 = config_scores.get("tool1", {}).get(m)
        t2 = config_scores.get("tool2", {}).get(m)
        both = config_scores.get("both", {}).get(m)
        p2_text = "N/A".rjust(17) if p2 is None else f"{p2:>16.2f}%"
        print(
            f"pass@1@{m:<4} | {p2_text} | "
            f"{_format_score(t1, p2)} | {_format_score(t2, p2)} | "
            f"{_format_score(both, p2)}"
        )
    print("=" * 100)


def _tool_configs_to_run(phase: str, tools: str = None) -> List[str]:
    if phase != "mcp":
        return []
    if tools is None:
        return ["tool1", "tool2", "both"]
    return [tools]


def run_experiment(
    phase: str,
    variant: str = None,
    limit: int = None,
    model: str = REPRODUCTION_MODEL,
    tools: str = None,
):
    """Run phase 1/2/3 experiments."""
    use_mcp = phase == "mcp"
    require_cache = phase == "mcp"
    tool_names = _tool_configs_to_run(phase, tools)

    if not os.path.exists("data/mbpp.jsonl"):
        print("MBPP dataset not found, downloading...")
        download_mbpp()

    examples = load_mbpp(limit=limit)
    print(f"Loaded {len(examples)} examples")

    if require_cache:
        ensure_phase_cache_exists(examples)

    if not require_cache:
        from config import USE_BATCH_API

        if USE_BATCH_API:
            from ticoder_core.generator import pre_generate_all_tasks

            pre_generate_all_tasks(examples, model=model)

    baseline = run_baseline(examples, model, require_cache=require_cache)
    variants_to_run = ["passfail", "output"] if variant is None else [variant]

    all_results = {
        "baseline": baseline,
        "model": model,
        "n_examples": len(examples),
        "n_suggestions": NUM_CODE_SUGGESTIONS,
        "use_mcp": use_mcp,
    }

    if use_mcp:
        all_results["tool_configs"] = tool_names
        phase3_scores_by_variant = {v: {} for v in variants_to_run}

        for tool_name in tool_names:
            config = TOOL_CONFIGS[tool_name]
            print(f"\n{'=' * 60}")
            print(
                f"Running {config['phase']}: {config['label']} "
                f"({config['description']})"
            )
            print(f"{'=' * 60}")

            for v in variants_to_run:
                print(f"\nRunning TICODER-{v.upper()} + {config['label']}")
                results_file = f"results/ticoder_{v}_mcp_{tool_name}.json"
                results = run_ticoder_on_dataset(
                    examples,
                    variant=v,
                    use_cache=True,
                    use_mcp=True,
                    use_tool1=config["use_tool1"],
                    use_tool2=config["use_tool2"],
                    model=model,
                    results_file=results_file,
                    require_cache=require_cache,
                )

                m_scores = compute_pass_at_k_at_m(results, k=1)
                key = f"ticoder_{v}_{tool_name}"
                all_results[key] = {
                    "m_scores": m_scores,
                    "results": results,
                }
                phase3_scores_by_variant[v][tool_name] = m_scores

                print(f"\nTICODER-{v.upper()} {config['label']} Results:")
                for m, score in m_scores.items():
                    print(f"  pass@1@{m}: {score:.2f}%")

                adaptive_stats = compute_adaptive_stopping_stats(results)
                all_results[f"{key}_adaptive_stats"] = adaptive_stats
                if config["use_tool1"]:
                    print(
                        f"\n  Adaptive stopping: avg m={adaptive_stats['avg_m']}, "
                        f"efficiency gain={adaptive_stats['efficiency_gain_pct']}%"
                    )

        for v in variants_to_run:
            _print_phase3_ablation_table(v, phase3_scores_by_variant[v])

    else:
        for v in variants_to_run:
            print(f"\n{'=' * 60}")
            print(f"Running TICODER-{v.upper()}")
            print(f"{'=' * 60}")

            results_file = f"results/ticoder_{v}_{phase}.json"
            results = run_ticoder_on_dataset(
                examples,
                variant=v,
                use_cache=True,
                use_mcp=False,
                model=model,
                results_file=results_file,
                require_cache=require_cache,
            )

            m_scores = compute_pass_at_k_at_m(results, k=1)
            all_results[f"ticoder_{v}"] = {"m_scores": m_scores, "results": results}

            print(f"\nTICODER-{v.upper()} Results:")
            for m, score in m_scores.items():
                print(f"  pass@1@{m}: {score:.2f}%")

        passfail_scores = all_results.get("ticoder_passfail", {}).get("m_scores", {})
        output_scores = all_results.get("ticoder_output", {}).get("m_scores", {})
        print_results_table(
            baseline["pass1"],
            baseline["pass_at_n"],
            passfail_scores,
            output_scores,
            model_name=model,
            dataset_name=f"MBPP ({len(examples)} examples)",
        )

    os.makedirs("results", exist_ok=True)
    summary = {
        "baseline": {
            "pass1": baseline["pass1"],
            "pass_at_n": baseline["pass_at_n"],
            "per_example": baseline["results"],
        },
        "model": model,
        "n_examples": len(examples),
        "use_mcp": use_mcp,
        "temperature": TEMPERATURE,
        "num_code_suggestions": NUM_CODE_SUGGESTIONS,
        "num_test_suggestions": NUM_TEST_SUGGESTIONS,
    }

    if use_mcp:
        summary["tool_configs"] = tool_names
        for v in variants_to_run:
            for tool_name in tool_names:
                key = f"ticoder_{v}_{tool_name}"
                if key not in all_results:
                    continue
                m_scores = all_results[key]["m_scores"]
                summary[key] = {
                    "m_scores": m_scores,
                    "abs_improvement": round(m_scores.get(5, 0) - baseline["pass1"], 2),
                }
                stats_key = f"{key}_adaptive_stats"
                if stats_key in all_results:
                    summary[stats_key] = all_results[stats_key]
    else:
        for v in variants_to_run:
            key = f"ticoder_{v}"
            if key in all_results:
                m_scores = all_results[key]["m_scores"]
                summary[key] = {
                    "m_scores": m_scores,
                    "abs_improvement": round(m_scores.get(5, 0) - baseline["pass1"], 2),
                }

    atomic_write_json(f"results/final_{phase}.json", summary, indent=2)

    print(f"\nFinal results saved to results/final_{phase}.json")
    return all_results


def _run_noise_config(
    examples: list,
    n_flips: int,
    variant: str,
    use_soft_scoring: bool,
    checkpoint_file: str,
    all_checkpoints: dict,
    completed: set,
):
    """Run one noise config (hard or soft) with checkpointing."""
    from mcp_tools.noise_recovery import run_noise_experiment_on_example

    config_key = f"{n_flips}_{'soft' if use_soft_scoring else 'hard'}"
    method = "soft scoring" if use_soft_scoring else "hard pruning"
    label = f"{n_flips}/5 flips ({method})"

    if config_key in completed:
        print(f"\n  Skipping {label} (already in checkpoint)")
        return all_checkpoints.get(config_key, [])

    print(f"\n{'=' * 60}")
    n_workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"Running: {label} ({n_workers} parallel workers)")
    print(f"{'=' * 60}")

    config_results = list(all_checkpoints.get(config_key, []))
    existing_task_ids = {str(r.get("task_id")) for r in config_results}
    tasks_to_run = [
        ex for ex in examples
        if str(ex["task_id"]) not in existing_task_ids
    ]
    save_lock = threading.Lock()
    counter = [0]
    total = len(tasks_to_run)

    if not tasks_to_run:
        print(f"  {label}: all {len(config_results)} tasks already checkpointed")
        all_checkpoints[config_key] = config_results
        completed.add(config_key)
        _save_noise_checkpoint(checkpoint_file, all_checkpoints, completed)
        return config_results

    def _run_one(example, _nf=n_flips, _v=variant, _ss=use_soft_scoring):
        return run_noise_experiment_on_example(
            example, n_flips=_nf, variant=_v, use_soft_scoring=_ss,
        )

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_one, ex): ex for ex in tasks_to_run}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                ex = futures[future]
                print(f"  Task {ex.get('task_id','?')} failed: {e}")
                continue

            if result:
                with save_lock:
                    config_results.append(result)
                    counter[0] += 1
                    if counter[0] % 50 == 0 or counter[0] == total:
                        print(f"  {label}: {counter[0]}/{total} tasks done")
                    all_checkpoints[config_key] = config_results
                    _save_noise_checkpoint(checkpoint_file, all_checkpoints, completed)

    all_checkpoints[config_key] = config_results
    completed.add(config_key)

    if config_results:
        m_scores = compute_pass_at_k_at_m(config_results, k=1)
        print(f"\n  Results for {label}:")
        for m, score in sorted(m_scores.items()):
            print(f"    pass@1@{m}: {score:.2f}%")

    _save_noise_checkpoint(checkpoint_file, all_checkpoints, completed)
    return config_results


def _save_noise_checkpoint(checkpoint_file, all_checkpoints, completed):
    """Save Phase 4 checkpoint to disk."""
    entries = [
        {
            "config_key": ck,
            "results": res,
            "complete": ck in completed,
        }
        for ck, res in all_checkpoints.items()
    ]
    atomic_write_json(checkpoint_file, entries, indent=2)


def run_noise_experiment(
    variant: str = "passfail",
    limit: int = None,
    model: str = REPRODUCTION_MODEL,
):
    """
    Run the noisy-oracle robustness experiment (Phase 4).

    For each noise level, runs two configs:
    - Hard pruning (no recovery): shows how noise degrades the paper's method
    - Soft scoring (Tool 2): shows how soft scoring absorbs noise

    Cache from Phase 2 is mandatory.
    """
    del model  # Phase 4 is cache-only and does not generate new LLM calls.

    from mcp_tools.noise_recovery import NOISE_LEVELS

    if not os.path.exists("data/mbpp.jsonl"):
        print("MBPP dataset not found, downloading...")
        download_mbpp()

    examples = load_mbpp(limit=limit)
    print(f"Loaded {len(examples)} examples for noise experiment")

    ensure_phase_cache_exists(examples)

    phase2_baseline_m5 = None
    phase2_file = "results/final_reproduce.json"
    if os.path.exists(phase2_file):
        with open(phase2_file, "r", encoding="utf-8") as f:
            phase2_data = json.load(f)
        variant_key = f"ticoder_{variant}"
        phase2_m_scores = phase2_data.get(variant_key, {}).get("m_scores", {})
        phase2_baseline_m5 = phase2_m_scores.get("5", phase2_m_scores.get(5))
        if phase2_baseline_m5 is None:
            print(
                "Warning: Could not find pass@1@5 in results/final_reproduce.json "
                f"for variant '{variant}'."
            )
    else:
        print(
            "Warning: results/final_reproduce.json not found. "
            "0% baseline reference will be omitted from the printed table."
        )

    checkpoint_file = "results/phase4_checkpoint.json"
    all_checkpoints: dict = {}
    completed: set = set()

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
        for entry in checkpoint_data:
            ck = entry["config_key"]
            all_checkpoints[ck] = entry["results"]
            if entry.get("complete", True) and len(entry["results"]) >= len(examples):
                completed.add(ck)
        print(f"Resumed from checkpoint: {len(completed)} configs already done")

    results_all = {}

    for n_flips in NOISE_LEVELS:
        # Run hard pruning (no recovery) first
        hard_results = _run_noise_config(
            examples, n_flips, variant, use_soft_scoring=False,
            checkpoint_file=checkpoint_file,
            all_checkpoints=all_checkpoints, completed=completed,
        )
        results_all[f"{n_flips}_hard"] = hard_results

        # Run soft scoring (with recovery)
        soft_results = _run_noise_config(
            examples, n_flips, variant, use_soft_scoring=True,
            checkpoint_file=checkpoint_file,
            all_checkpoints=all_checkpoints, completed=completed,
        )
        results_all[f"{n_flips}_soft"] = soft_results

    # Print comparison table
    print(f"\n{'=' * 80}")
    print("PHASE 4: NOISE ROBUSTNESS COMPARISON")
    print(f"{'=' * 80}")
    print(f"{'Metric':<12} | {'Phase 2 (0 flips)':>18} | {'1 flip + hard':>18} | {'1 flip + soft':>18}")
    print("-" * 80)

    phase2_m_scores = {}
    if os.path.exists(phase2_file):
        with open(phase2_file, "r", encoding="utf-8") as f:
            phase2_data = json.load(f)
        phase2_m_scores = phase2_data.get(f"ticoder_{variant}", {}).get("m_scores", {})

    for m in range(1, 6):
        p2 = phase2_m_scores.get(str(m), phase2_m_scores.get(m))
        hard_scores = compute_pass_at_k_at_m(results_all.get("1_hard", []), k=1)
        soft_scores = compute_pass_at_k_at_m(results_all.get("1_soft", []), k=1)
        h = hard_scores.get(m)
        s = soft_scores.get(m)
        p2_text = f"{p2:>17.2f}%" if p2 is not None else "N/A".rjust(18)
        h_text = f"{h:>8.2f}% ({h - p2:+6.2f})" if h is not None and p2 is not None else "N/A".rjust(18)
        s_text = f"{s:>8.2f}% ({s - p2:+6.2f})" if s is not None and p2 is not None else "N/A".rjust(18)
        print(f"pass@1@{m:<4} | {p2_text} | {h_text} | {s_text}")
    print(f"{'=' * 80}")

    # Save final results
    os.makedirs("results", exist_ok=True)
    noise_summary = {}

    for config_key, results in results_all.items():
        m_scores = compute_pass_at_k_at_m(results, k=1)
        total_flips = sum(r.get("noise_flips", 0) for r in results)
        noise_summary[config_key] = {
            "config_key": config_key,
            "m_scores": m_scores,
            "total_noise_flips": total_flips,
            "n_examples": len(results),
        }

    noise_summary["phase2_reference_0_flips"] = {
        "description": "Copied baseline from Phase 2 (0/5 flips, no new run)",
        "variant": variant,
        "pass_at_1_at_5": phase2_baseline_m5,
    }

    atomic_write_json("results/final_noisy.json", noise_summary, indent=2)

    print(f"\nNoise results saved to results/final_noisy.json")
    return results_all


def main():
    parser = argparse.ArgumentParser(description="Run TICODER experiments")
    parser.add_argument(
        "--phase",
        choices=["pilot", "reproduce", "mcp", "noisy"],
        default="pilot",
        help="Experiment phase",
    )
    parser.add_argument(
        "--variant",
        choices=["passfail", "output", "both"],
        default="both",
        help="TICODER variant to run",
    )
    parser.add_argument(
        "--tools",
        choices=["tool1", "tool2", "both"],
        default="both",
        help=(
            "Which tools to enable in Phase 3: tool1 (entropy only), "
            "tool2 (soft scoring only), both. Omit to run all ablations."
        ),
    )
    parser.add_argument("--model", default=REPRODUCTION_MODEL, help="Claude model to use")
    args = parser.parse_args()

    tools_was_specified = any(
        arg == "--tools" or arg.startswith("--tools=") for arg in sys.argv[1:]
    )
    selected_tools = args.tools if tools_was_specified else None

    if args.phase == "pilot":
        limit = PILOT_SIZE
        print(f"\n--- PHASE 1: PILOT TEST ({limit} examples)")
    elif args.phase == "reproduce":
        limit = FULL_SIZE
        print("\n--- PHASE 2: FULL REPRODUCTION")
    elif args.phase == "mcp":
        limit = FULL_SIZE
        if selected_tools is None:
            print("\n--- PHASE 3: MCP ABLATION (Tool 1, Tool 2, Both)")
        else:
            config = TOOL_CONFIGS[selected_tools]
            print(
                f"\n--- PHASE 3: MCP EVALUATION ({config['label']}: "
                f"{config['description']})"
            )
    else:
        limit = FULL_SIZE
        print("\n--- PHASE 4: NOISY ORACLE EXPERIMENT")

    variant = None if args.variant == "both" else args.variant

    if args.phase == "noisy":
        run_noise_experiment(variant=variant or "passfail", limit=limit, model=args.model)
    else:
        run_experiment(args.phase, variant, limit, args.model, tools=selected_tools)


if __name__ == "__main__":
    main()
