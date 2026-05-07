"""
ticoder_core/generator.py
Generates code and test suggestions using Anthropic API.
"""

import json
import os
import sys
import time
from typing import List

import anthropic

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ANTHROPIC_API_KEY,
    MAX_TOKENS,
    NUM_CODE_SUGGESTIONS,
    NUM_TEST_SUGGESTIONS,
    REPRODUCTION_MODEL,
    TEMPERATURE,
    USE_BATCH_API,
)
from utils.oracle import parse_test_assertion
from utils.prompts import build_code_prompt, build_test_prompt, extract_func_name
from utils.safe_io import atomic_write_json

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
PROMPT_FORMAT_VERSION = "ticoder-paper-system-backtick-tags-v1"


# ----------------------
# Low-level API helpers
# ----------------------

def generate_single(
    prompt: str,
    model: str,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    system: str = None,
) -> str:
    """Generate one completion via direct Messages API."""
    try:
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
    except anthropic.RateLimitError:
        print("Rate limit hit, waiting 60 seconds...")
        time.sleep(60)
        return generate_single(
            prompt,
            model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
        )
    except Exception as exc:
        print(f"Generation error: {exc}")
        return ""


def _supports_batch_api() -> bool:
    return bool(getattr(getattr(client, "messages", None), "batches", None))


def _configured_api_type() -> str:
    if USE_BATCH_API and _supports_batch_api():
        return "anthropic_messages_batch"
    return "anthropic_messages_direct"


def _extract_status(batch_obj) -> str:
    for attr in ("processing_status", "status"):
        value = getattr(batch_obj, attr, None)
        if value:
            return str(value)
    if isinstance(batch_obj, dict):
        return str(batch_obj.get("processing_status") or batch_obj.get("status") or "")
    return ""


def _extract_batch_id(batch_obj) -> str:
    if isinstance(batch_obj, dict):
        return batch_obj.get("id", "")
    return getattr(batch_obj, "id", "")


def _load_json_if_exists(path: str, default=None):
    if not path or not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_token(value) -> str:
    import re

    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return token.strip("._") or "state"


def _pregen_state_path(
    model: str,
    n_codes: int,
    n_tests: int,
    cache_dir: str,
) -> str:
    run_name = _safe_token(
        f"{model}_{n_codes}codes_{n_tests}tests_{MAX_TOKENS}tokens_"
        f"temp{TEMPERATURE}_{PROMPT_FORMAT_VERSION}"
    )
    cache_name = _safe_token(os.path.abspath(cache_dir))
    return os.path.join("results", "batch_state", f"pregen_{cache_name}_{run_name}.json")


def _pregen_metadata(model: str, n_codes: int, n_tests: int, cache_dir: str) -> dict:
    return {
        "model": model,
        "n_codes": n_codes,
        "n_tests": n_tests,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "prompt_format_version": PROMPT_FORMAT_VERSION,
        "cache_dir": os.path.abspath(cache_dir),
        "api_type": _configured_api_type(),
    }


def _pregen_state_matches(state: dict, metadata: dict) -> bool:
    return bool(state and state.get("metadata") == metadata)


def _as_iterable_results(results_obj):
    if results_obj is None:
        return []
    if isinstance(results_obj, list):
        return results_obj
    if isinstance(results_obj, dict):
        if isinstance(results_obj.get("data"), list):
            return results_obj["data"]
        return []
    data = getattr(results_obj, "data", None)
    if isinstance(data, list):
        return data
    try:
        return list(results_obj)
    except Exception:
        return []


def _extract_result_text(result_item) -> str:
    """
    Best-effort extraction for batch result entries across SDK object shapes.
    Returns empty string if entry is not successful or unparsable.
    """
    item = result_item

    if isinstance(item, dict):
        result = item.get("result")
        if not isinstance(result, dict):
            return ""
        if result.get("type") != "succeeded":
            return ""
        message = result.get("message", {})
        content = message.get("content", [])
        if content and isinstance(content[0], dict):
            return str(content[0].get("text", "")).strip()
        return ""

    result = getattr(item, "result", None)
    if result is None:
        return ""

    r_type = getattr(result, "type", None)
    if r_type and r_type != "succeeded":
        return ""

    message = getattr(result, "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", None)
    if not content:
        return ""

    first = content[0]
    text = getattr(first, "text", None)
    return str(text).strip() if text else ""


def generate_batch(
    prompt: str,
    model: str,
    n: int,
    max_tokens: int,
    temperature: float,
    system: str = None,
    poll_interval_sec: int = 5,
    timeout_sec: int = 600,
) -> List[str]:
    """
    Attempt Anthropic Message Batches API. Falls back to sequential on failure.
    """
    if n <= 0:
        return []

    if not USE_BATCH_API or not _supports_batch_api():
        return [
            generate_single(
                prompt,
                model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
            )
            for _ in range(n)
        ]

    try:
        requests = []
        for i in range(n):
            requests.append(
                {
                    "custom_id": f"req-{i}",
                    "params": {
                        "model": model,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "system": system or "",
                        "messages": [{"role": "user", "content": prompt}],
                    },
                }
            )

        batch = client.messages.batches.create(requests=requests)
        batch_id = _extract_batch_id(batch)
        if not batch_id:
            raise RuntimeError("Batch creation did not return an id")

        start = time.time()
        while True:
            current = client.messages.batches.retrieve(batch_id)
            status = _extract_status(current)
            if status in {"ended", "completed", "succeeded"}:
                break
            if status in {"failed", "expired", "cancelled"}:
                raise RuntimeError(f"Batch ended with status: {status}")
            if (time.time() - start) > timeout_sec:
                raise TimeoutError(f"Batch polling timed out after {timeout_sec} seconds")
            time.sleep(poll_interval_sec)

        raw_results = client.messages.batches.results(batch_id)
        entries = _as_iterable_results(raw_results)
        texts = [t for t in (_extract_result_text(entry) for entry in entries) if t]

        # If API returned fewer successful results, backfill sequentially.
        while len(texts) < n:
            texts.append(
                generate_single(
                    prompt,
                    model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                )
            )

        return texts[:n]

    except Exception as exc:
        print(f"Batch generation failed ({exc}); falling back to sequential calls.")
        return [
            generate_single(
                prompt,
                model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
            )
            for _ in range(n)
        ]


# ----------------------
# Normalization helpers
# ----------------------

def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences (```python ... ```) from LLM output."""
    import re
    # Remove opening fence: ```python or ```
    text = re.sub(r'^```(?:python)?\s*\n?', '', text, flags=re.MULTILINE)
    # Remove closing fence: ```
    text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
    return text.strip()


def _extract_code_tags(text: str) -> str:
    """Extract content wrapped by paper/GitHub delimiters if present."""
    import re

    match = re.search(r'<code>(.*?)</code>', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # The Microsoft TiCoder repo asks for single-backtick tags. Avoid
    # treating Markdown triple-fence output as single-backtick wrapping.
    if "```" not in text:
        match = re.search(r'(?<!`)`(?!`)(.*?)(?<!`)`(?!`)', text, re.DOTALL)
        if match:
            return match.group(1).strip()

    return text


def _normalize_code_candidate(raw: str, func_header: str, description: str) -> str:
    text = _extract_code_tags((raw or "").strip())
    text = _strip_markdown_fences(text)
    if not text:
        return ""

    if text.startswith("def "):
        return text

    return f"{func_header}\n    \"\"\"{description}\"\"\"\n{text}"


def _keep_first_assert_comparison(candidate: str, func_name: str) -> str:
    """If an assert chains comparisons with `and`, keep the first function check."""
    try:
        import ast

        tree = ast.parse(candidate)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
            return candidate

        expr = tree.body[0].test
        if not isinstance(expr, ast.BoolOp) or not isinstance(expr.op, ast.And):
            return candidate

        for value in expr.values:
            text = f"assert {ast.unparse(value)}"
            if f"{func_name}(" in text and "==" in text:
                return text
    except Exception:
        return candidate

    return candidate


def _normalize_test_candidate(raw: str, func_name: str) -> str:
    text = _extract_code_tags((raw or "").strip())
    text = _strip_markdown_fences(text)
    if not text:
        return ""

    # Find the first assert line (LLM may wrap in a test function).
    candidate_text = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("assert "):
            candidate_text = line
            break
        if line.startswith(f"{func_name}(") and "==" in line:
            candidate_text = line
            break

    # Fallback: use first non-empty line.
    if candidate_text is None:
        for line in text.splitlines():
            line = line.strip()
            if line:
                candidate_text = line
                break

    if not candidate_text:
        return ""

    if candidate_text.startswith("assert "):
        candidate = candidate_text
    elif candidate_text.startswith(f"{func_name}("):
        candidate = f"assert {candidate_text}"
    else:
        candidate = f"assert {func_name}({candidate_text}"

    candidate = _keep_first_assert_comparison(candidate, func_name)

    # Basic sanity checks.
    if "==" not in candidate:
        return ""
    if f"{func_name}(" not in candidate:
        return ""

    parsed_input, _ = parse_test_assertion(candidate)
    if parsed_input is None:
        return ""

    return candidate


def _dedupe_keep_order(items: List[str]) -> List[str]:
    """Simple exact-string dedup. Used for tests only."""
    seen = set()
    out = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _behavioral_dedupe_codes(
    codes: List[str],
    func_header: str,
    max_probes: int = 3,
) -> List[str]:
    """
    Keep codes that are behaviourally distinct.

    Two codes are duplicates only if they produce identical outputs on
    every probe input.  Syntactically different code that behaves
    differently is always kept.

    Probe inputs are lightweight: small representative values chosen by
    type-hints in the header, falling back to generic integers/strings.
    """
    from utils.oracle import execute_code

    if not codes:
        return codes

    # Build a small set of probe inputs from the function signature.
    func_name = extract_func_name(func_header)
    probe_inputs = _make_probe_inputs(func_header, max_probes)

    seen_signatures: List[tuple] = []
    kept: List[str] = []

    for code in codes:
        sig = _compute_behavior_signature(code, func_name, probe_inputs)
        if sig not in seen_signatures:
            seen_signatures.append(sig)
            kept.append(code)

    return kept


def _make_probe_inputs(func_header: str, max_probes: int = 3) -> List[str]:
    """
    Generate small probe-input strings for behavioural dedup.

    Inspects the function parameter list to guess reasonable tiny inputs.
    Falls back to generic integer/string probes when heuristics fail.
    """
    import re

    # Extract parameter list: "def foo(a, b, c=5):" → "a, b, c=5"
    m = re.search(r'\((.*?)\)', func_header)
    if not m:
        return ["0", "1", "-1"]

    param_str = m.group(1)
    params = [p.strip().split("=")[0].strip() for p in param_str.split(",") if p.strip()]
    n_params = len(params)

    if n_params == 0:
        return [""]  # no-arg function

    # A few diverse probe sets per arity
    if n_params == 1:
        return ["[1, 2, 3]", "0", "'abc'"][:max_probes]
    elif n_params == 2:
        return ["[1, 2, 3], [2, 3, 4]", "1, 2", "'hello', 'lo'"][:max_probes]
    elif n_params == 3:
        return ["1, 2, 3", "[1,2], [3,4], 2", "'ab', 'cd', 1"][:max_probes]
    else:
        # Generic: fill with small ints
        args = ", ".join(str(i) for i in range(1, n_params + 1))
        return [args]


def _compute_behavior_signature(
    code: str, func_name: str, probe_inputs: List[str]
) -> tuple:
    """Run code on each probe input and return a tuple of outputs."""
    from utils.oracle import execute_code

    CRASH = "__CRASH__"
    TIMEOUT = "__TIMEOUT__"
    sig = []
    for inp in probe_inputs:
        success, output, error = execute_code(code, inp, timeout=2)
        if not success:
            sig.append(TIMEOUT if "timed out" in (error or "") else CRASH)
        else:
            try:
                # Make hashable
                if isinstance(output, list):
                    sig.append(("list", tuple(output)))
                elif isinstance(output, dict):
                    sig.append(("dict", tuple(sorted(output.items()))))
                elif isinstance(output, set):
                    sig.append(("set", tuple(sorted(output))))
                else:
                    sig.append(output)
            except Exception:
                sig.append(str(output))
    return tuple(sig)


# ----------------------
# Public generation APIs
# ----------------------

def generate_code_suggestions(
    description: str,
    func_header: str,
    n: int = NUM_CODE_SUGGESTIONS,
    model: str = REPRODUCTION_MODEL,
) -> List[str]:
    """Generate `n` code suggestions for one problem."""
    prompt_parts = build_code_prompt(description, func_header)

    print(f"  Generating {n} code suggestions...")
    raw_outputs = generate_batch(
        prompt=prompt_parts["user"],
        model=model,
        n=n,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=prompt_parts["system"],
    )

    normalised = [_normalize_code_candidate(raw, func_header, description)
                  for raw in raw_outputs]
    # Remove empty/broken outputs first, then exact-string dedupe.
    # NOTE: We intentionally keep behaviourally similar codes.
    # The paper uses all N suggestions; TICODER's test-based pruning
    # handles behavioural differences — aggressive pre-dedup destroys
    # the candidate pool (avg 2 codes/task instead of ~25).
    valid = [c for c in normalised if c]
    suggestions = _dedupe_keep_order(valid)

    print(f"  Generated {len(suggestions)} unique code suggestions "
          f"(from {len(valid)} valid, {n} raw)")
    return suggestions


def generate_test_suggestions(
    description: str,
    func_header: str,
    n: int = NUM_TEST_SUGGESTIONS,
    model: str = REPRODUCTION_MODEL,
) -> List[str]:
    """Generate `n` test assertion suggestions for one problem."""
    func_name = extract_func_name(func_header)
    prompt_parts = build_test_prompt(description, func_header, func_name)

    print(f"  Generating {n} test suggestions...")
    raw_outputs = generate_batch(
        prompt=prompt_parts["user"],
        model=model,
        n=n,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=prompt_parts["system"],
    )

    suggestions = _dedupe_keep_order(
        [_normalize_test_candidate(raw, func_name) for raw in raw_outputs]
    )

    print(f"  Generated {len(suggestions)} unique test suggestions")
    return suggestions


def cache_generations(
    task_id: str,
    code_suggestions: List[str],
    test_suggestions: List[str],
    cache_dir: str = "cache",
    metadata: dict = None,
):
    """Save generated suggestions to cache."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{task_id}.json")
    atomic_write_json(
        cache_file,
        {
            "task_id": task_id,
            "code_suggestions": code_suggestions,
            "test_suggestions": test_suggestions,
            "generation_metadata": metadata or {
                "api_type": _configured_api_type(),
                "prompt_format_version": PROMPT_FORMAT_VERSION,
                "temperature": TEMPERATURE,
                "num_code_suggestions_config": NUM_CODE_SUGGESTIONS,
                "num_test_suggestions_config": NUM_TEST_SUGGESTIONS,
                "max_tokens": MAX_TOKENS,
            },
        },
        indent=2,
    )


def load_from_cache(task_id: str, cache_dir: str = "cache"):
    """Load cached generations for task id if available."""
    cache_file = os.path.join(cache_dir, f"{task_id}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ──────────────────────────────────────────────────────────────────────
# Mega-batch: pre-generate ALL tasks in 1–2 large Batch API submissions
# ──────────────────────────────────────────────────────────────────────

MAX_BATCH_SIZE = 10_000  # Conservative chunk size; Anthropic allows larger batches.


def pre_generate_all_tasks(
    examples: list,
    model: str = REPRODUCTION_MODEL,
    n_codes: int = NUM_CODE_SUGGESTIONS,
    n_tests: int = NUM_TEST_SUGGESTIONS,
    cache_dir: str = "cache",
):
    """
    Submit ALL code and test prompts for every task as mega-batches,
    poll until done, then normalise, dedupe and cache results per task.

    Skips tasks that already have a complete cache entry.
    """
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.join("results", "batch_state"), exist_ok=True)
    metadata = _pregen_metadata(model, n_codes, n_tests, cache_dir)
    state_file = _pregen_state_path(model, n_codes, n_tests, cache_dir)
    state = _load_json_if_exists(state_file, default={})
    code_texts = None
    test_texts = None

    # ── 1. Build prompt lists, skipping cached tasks ──────────────
    if _pregen_state_matches(state, metadata) and state.get("tasks_to_generate"):
        tasks_to_generate = [tuple(item) for item in state["tasks_to_generate"]]
        code_texts = state.get("code_texts")
        test_texts = state.get("test_texts")
        print(
            f"Resuming mega-batch checkpoint: {len(tasks_to_generate)} tasks "
            f"from {state_file}"
        )
    else:
        tasks_to_generate = []   # [(task_id, description, func_header, func_name)]

        for ex in examples:
            task_id = str(ex["task_id"])
            cached = load_from_cache(task_id, cache_dir)
            if (cached
                    and cached.get("code_suggestions")
                    and cached.get("test_suggestions")):
                continue  # already cached

            func_header = ""
            for line in ex["code"].split("\n"):
                if line.strip().startswith("def "):
                    func_header = line.strip()
                    break

            func_name = extract_func_name(func_header)
            tasks_to_generate.append((task_id, ex["text"], func_header, func_name))

        state = {
            "metadata": metadata,
            "status": "tasks_built",
            "tasks_to_generate": [list(item) for item in tasks_to_generate],
            "code_texts": None,
            "test_texts": None,
        }
        atomic_write_json(state_file, state, indent=2)

    if not tasks_to_generate:
        print("All tasks already cached — nothing to generate.")
        return

    total_code_reqs = len(tasks_to_generate) * n_codes
    total_test_reqs = len(tasks_to_generate) * n_tests
    print(f"\nMega-batch: {len(tasks_to_generate)} tasks to generate "
          f"({total_code_reqs} code + {total_test_reqs} test requests)")
    print(f"  API mode: {_configured_api_type()} via Anthropic Messages API")

    # ── 2. Build batch request lists ──────────────────────────────
    code_requests = []
    test_requests = []

    for idx, (task_id, description, func_header, func_name) in enumerate(tasks_to_generate):
        code_prompt = build_code_prompt(description, func_header)
        test_prompt = build_test_prompt(description, func_header, func_name)

        for j in range(n_codes):
            code_requests.append({
                "custom_id": f"code-{idx}-{j}",
                "params": {
                    "model": model,
                    "max_tokens": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                    "system": code_prompt["system"],
                    "messages": [{"role": "user", "content": code_prompt["user"]}],
                },
            })

        for j in range(n_tests):
            test_requests.append({
                "custom_id": f"test-{idx}-{j}",
                "params": {
                    "model": model,
                    "max_tokens": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                    "system": test_prompt["system"],
                    "messages": [{"role": "user", "content": test_prompt["user"]}],
                },
            })

    # ── 3. Submit & poll batches ──────────────────────────────────
    if not code_texts or len(code_texts) != total_code_reqs:
        code_texts = _submit_and_collect_mega_batch(
            code_requests,
            label="CODE",
            state_dir=os.path.join("results", "batch_state"),
            run_id=_safe_token(os.path.basename(state_file)),
        )
        state["code_texts"] = code_texts
        state["status"] = "code_done"
        atomic_write_json(state_file, state, indent=2)
    else:
        print("  [CODE] Reusing completed texts from mega-batch checkpoint.")

    if not test_texts or len(test_texts) != total_test_reqs:
        test_texts = _submit_and_collect_mega_batch(
            test_requests,
            label="TEST",
            state_dir=os.path.join("results", "batch_state"),
            run_id=_safe_token(os.path.basename(state_file)),
        )
        state["test_texts"] = test_texts
        state["status"] = "test_done"
        atomic_write_json(state_file, state, indent=2)
    else:
        print("  [TEST] Reusing completed texts from mega-batch checkpoint.")

    # ── 4. Distribute results back to tasks, normalise & cache ────
    code_ptr = 0
    test_ptr = 0

    for idx, (task_id, description, func_header, func_name) in enumerate(tasks_to_generate):
        cached = load_from_cache(task_id, cache_dir)
        if (cached
                and cached.get("code_suggestions")
                and cached.get("test_suggestions")):
            code_ptr += n_codes
            test_ptr += n_tests
            print(f"  [{idx + 1}/{len(tasks_to_generate)}] Task {task_id}: already cached")
            continue

        raw_codes = code_texts[code_ptr: code_ptr + n_codes]
        raw_tests = test_texts[test_ptr: test_ptr + n_tests]
        code_ptr += n_codes
        test_ptr += n_tests

        # Normalise codes (exact-string dedup only — preserve behavioural diversity)
        normalised = [_normalize_code_candidate(r, func_header, description)
                      for r in raw_codes]
        valid_codes = [c for c in normalised if c]
        codes = _dedupe_keep_order(valid_codes)

        # Normalise tests
        tests = _dedupe_keep_order(
            [_normalize_test_candidate(r, func_name) for r in raw_tests]
        )

        cache_generations(task_id, codes, tests, cache_dir, metadata=metadata)
        print(f"  [{idx + 1}/{len(tasks_to_generate)}] Task {task_id}: "
              f"{len(codes)} codes, {len(tests)} tests cached")

    state["status"] = "complete"
    atomic_write_json(state_file, state, indent=2)
    print(f"\nMega-batch complete: {len(tasks_to_generate)} tasks cached.")


def _submit_and_collect_mega_batch(
    requests: list,
    label: str = "",
    poll_interval_sec: int = 30,
    timeout_sec: int = 86400,     # Message Batches can take up to 24 hours.
    state_dir: str = None,
    run_id: str = None,
) -> List[str]:
    """
    Submit requests in chunks of MAX_BATCH_SIZE, poll until each
    completes, and return the concatenated text results in order.

    Falls back to sequential calls if the Batch API is unavailable.
    """
    if not requests:
        return []

    if state_dir:
        os.makedirs(state_dir, exist_ok=True)

    if not USE_BATCH_API or not _supports_batch_api():
        print(f"  [{label}] Batch API unavailable — using sequential calls "
              f"({len(requests)} requests)...")
        results = []
        for i, req in enumerate(requests):
            p = req["params"]
            text = generate_single(
                p["messages"][0]["content"],
                p["model"],
                max_tokens=p.get("max_tokens", MAX_TOKENS),
                temperature=p.get("temperature", TEMPERATURE),
                system=p.get("system"),
            )
            results.append(text)
            if (i + 1) % 100 == 0:
                print(f"    [{label}] {i + 1}/{len(requests)} done")
        return results

    # Split into chunks of MAX_BATCH_SIZE
    chunks = [requests[i:i + MAX_BATCH_SIZE]
              for i in range(0, len(requests), MAX_BATCH_SIZE)]

    all_texts: List[str] = []

    for chunk_idx, chunk in enumerate(chunks):
        chunk_label = f"{label} batch {chunk_idx + 1}/{len(chunks)}"
        print(f"  [{chunk_label}] Submitting {len(chunk)} requests...")
        state_path = None
        chunk_state = {}
        if state_dir:
            state_name = _safe_token(
                f"{run_id or 'run'}_{label}_{chunk_idx + 1}_of_{len(chunks)}"
            )
            state_path = os.path.join(state_dir, f"{state_name}.json")
            chunk_state = _load_json_if_exists(state_path, default={}) or {}

        try:
            batch_id = chunk_state.get("batch_id")
            if batch_id:
                print(f"  [{chunk_label}] Resuming Batch ID: {batch_id}")
            else:
                batch = client.messages.batches.create(requests=chunk)
                batch_id = _extract_batch_id(batch)
                if not batch_id:
                    raise RuntimeError("No batch id returned")
                chunk_state = {
                    "run_id": run_id,
                    "label": label,
                    "chunk_index": chunk_idx,
                    "n_chunks": len(chunks),
                    "batch_id": batch_id,
                    "status": "submitted",
                    "request_count": len(chunk),
                    "custom_ids": [req["custom_id"] for req in chunk],
                    "api_type": "anthropic_messages_batch",
                }
                if state_path:
                    atomic_write_json(state_path, chunk_state, indent=2)

            print(f"  [{chunk_label}] Batch ID: {batch_id} — polling...")
            start = time.time()
            while True:
                current = client.messages.batches.retrieve(batch_id)
                status = _extract_status(current)

                if status in {"ended", "completed", "succeeded"}:
                    elapsed = int(time.time() - start)
                    print(f"  [{chunk_label}] Done in {elapsed}s")
                    chunk_state["status"] = status
                    chunk_state["ended_at_local"] = int(time.time())
                    if state_path:
                        atomic_write_json(state_path, chunk_state, indent=2)
                    break
                if status in {"failed", "expired", "cancelled"}:
                    chunk_state["status"] = status
                    if state_path:
                        atomic_write_json(state_path, chunk_state, indent=2)
                    raise RuntimeError(f"Batch {batch_id} ended: {status}")
                if (time.time() - start) > timeout_sec:
                    chunk_state["status"] = status or "timeout"
                    if state_path:
                        atomic_write_json(state_path, chunk_state, indent=2)
                    raise TimeoutError(
                        f"Batch {batch_id} timed out after {timeout_sec}s")

                elapsed = int(time.time() - start)
                if elapsed % 60 < poll_interval_sec:
                    print(f"    ... still processing ({elapsed}s elapsed, "
                          f"status={status})")
                time.sleep(poll_interval_sec)

            # Collect results
            raw_results = client.messages.batches.results(batch_id)
            entries = _as_iterable_results(raw_results)

            # Build an {custom_id → text} map so order is preserved
            id_to_text = {}
            for entry in entries:
                cid = (entry.get("custom_id") if isinstance(entry, dict)
                       else getattr(entry, "custom_id", None))
                txt = _extract_result_text(entry)
                if cid is not None:
                    id_to_text[cid] = txt

            # Rebuild in original request order
            chunk_texts = [id_to_text.get(r["custom_id"], "") for r in chunk]

            # Backfill missing results sequentially
            missing = sum(1 for t in chunk_texts if not t)
            if missing:
                print(f"  [{chunk_label}] Backfilling {missing} missing "
                      f"results sequentially...")
                for i, t in enumerate(chunk_texts):
                    if not t:
                        p = chunk[i]["params"]
                        chunk_texts[i] = generate_single(
                            p["messages"][0]["content"],
                            p["model"],
                            max_tokens=p.get("max_tokens", MAX_TOKENS),
                            temperature=p.get("temperature", TEMPERATURE),
                            system=p.get("system"),
                        )

            chunk_state["status"] = "results_collected"
            chunk_state["missing_backfilled"] = missing
            if state_path:
                atomic_write_json(state_path, chunk_state, indent=2)
            all_texts.extend(chunk_texts)

        except Exception as exc:
            print(f"  [{chunk_label}] Failed ({exc}) — falling back to "
                  f"sequential for this chunk ({len(chunk)} requests)...")
            chunk_texts = chunk_state.get("sequential_texts")
            if not chunk_texts or len(chunk_texts) != len(chunk):
                chunk_texts = [""] * len(chunk)
            for i, req in enumerate(chunk):
                if chunk_texts[i]:
                    continue
                p = req["params"]
                text = generate_single(
                    p["messages"][0]["content"],
                    p["model"],
                    max_tokens=p.get("max_tokens", MAX_TOKENS),
                    temperature=p.get("temperature", TEMPERATURE),
                    system=p.get("system"),
                )
                chunk_texts[i] = text
                chunk_state["status"] = "sequential_fallback"
                chunk_state["sequential_texts"] = chunk_texts
                if state_path:
                    atomic_write_json(state_path, chunk_state, indent=2)
                if (i + 1) % 100 == 0:
                    print(f"    [{chunk_label}] fallback {i + 1}/{len(chunk)}")
            all_texts.extend(chunk_texts)

    return all_texts
