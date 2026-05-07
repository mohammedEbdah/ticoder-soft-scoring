"""
utils/oracle.py
Simulates user feedback using the reference implementation as an oracle.
This is exactly how the paper evaluates TICODER at scale (Section VII-D).

The oracle answers:
- PASS: test is consistent with user intent
- FAIL: test is inconsistent with user intent
- UNDEFINED: test input causes reference code to crash (precondition violation)
And for TICODER-OUTPUT: provides the correct output value.

Uses multiprocessing (not threading) so that infinite loops in generated
code can be reliably terminated via Process.terminate() on all platforms.

IMPORTANT: We use q.get(timeout=...) BEFORE p.join() to avoid the
Windows pipe-buffer race condition where get_nowait() returns Empty
even though the subprocess has already put() its result.
"""

from multiprocessing import Process, Queue as MPQueue
from typing import Tuple
import contextlib
import io


# Response constants (matching paper terminology)
PASS = "PASS"
FAIL = "FAIL"
UNDEFINED = "UNDEFINED"


# ── Multiprocessing worker functions ──────────────────────────────────
# Must be defined at module top-level so they are picklable on Windows
# (which uses the 'spawn' start method for multiprocessing).


def _execute_code_worker(code: str, test_input: str, result_queue):
    """Run a single function call in a subprocess."""
    try:
        namespace = {}
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(code, namespace)

            func_name = None
            for key, val in namespace.items():
                if callable(val) and not key.startswith("_"):
                    func_name = key
                    break

            if func_name is None:
                result_queue.put((False, None, "No function found in code"))
                return

            output = eval(f"{func_name}({test_input})", namespace)
        result_queue.put((True, output, ""))
    except Exception as e:
        result_queue.put((False, None, str(e)))


def _evaluate_correctness_worker(code: str, hidden_tests: list, result_queue):
    """Run all hidden tests against a code suggestion in a subprocess."""
    try:
        namespace = {}
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(code, namespace)

            for test in hidden_tests:
                try:
                    exec(test, namespace)
                except AssertionError:
                    result_queue.put(False)
                    return
                except Exception:
                    result_queue.put(False)
                    return

        result_queue.put(True)
    except Exception:
        result_queue.put(False)


# ── Public API ────────────────────────────────────────────────────────


def execute_code(code: str, test_input: str, timeout: int = 5) -> Tuple[bool, any, str]:
    """
    Execute a function with given input and return (success, output, error).

    Uses multiprocessing to properly terminate infinite loops on all
    platforms.  Reads result from Queue BEFORE joining the process to
    avoid Windows pipe-buffer race conditions.
    """
    q = MPQueue()
    p = Process(target=_execute_code_worker, args=(code, test_input, q))
    p.start()
    try:
        result = q.get(timeout=timeout)
    except Exception:
        result = (False, None, f"Execution timed out after {timeout}s")
    finally:
        if p.is_alive():
            p.terminate()
        p.join(timeout=5)
    return result


def parse_test_assertion(test_str: str) -> Tuple[str, any]:
    """
    Parse an assertion like 'assert func(input) == output'
    Returns (input_str, expected_output) or (None, None) if unparseable.
    """
    try:
        import ast

        tree = ast.parse(test_str.strip())
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
            raise ValueError("not a single assert")

        expr = tree.body[0].test
        if isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.And):
            expr = expr.values[0]

        if (
            not isinstance(expr, ast.Compare)
            or len(expr.ops) != 1
            or not isinstance(expr.ops[0], ast.Eq)
            or len(expr.comparators) != 1
            or not isinstance(expr.left, ast.Call)
        ):
            raise ValueError("not a simple equality comparison")

        call = expr.left
        args = [ast.unparse(arg) for arg in call.args]
        for kw in call.keywords:
            if kw.arg is None:
                args.append(f"**{ast.unparse(kw.value)}")
            else:
                args.append(f"{kw.arg}={ast.unparse(kw.value)}")

        expected = eval(ast.unparse(expr.comparators[0]))
        return ", ".join(args), expected

    except Exception:
        pass

    try:
        # Handle: assert func(input) == output
        if "==" in test_str:
            parts = test_str.split("==")
            lhs = parts[0].strip().replace("assert ", "")
            rhs = "==".join(parts[1:]).strip()

            # Extract input from function call
            # e.g. func([1,2,3]) -> [1,2,3]
            start = lhs.index("(") + 1
            end = lhs.rindex(")")
            input_str = lhs[start:end]

            # Parse expected output
            expected = eval(rhs)
            return input_str, expected

    except Exception:
        pass

    return None, None


def oracle_response(reference_code: str, test_str: str,
                    variant: str = "passfail") -> dict:
    """
    Simulate user response to a test using the reference implementation.
    """
    input_str, expected_output = parse_test_assertion(test_str)

    if input_str is None:
        return {"response": UNDEFINED, "correct_output": None, "input_str": None}

    success, actual_output, error = execute_code(reference_code, input_str)

    if not success:
        return {"response": UNDEFINED, "correct_output": None, "input_str": input_str}

    if actual_output == expected_output:
        return {"response": PASS, "correct_output": actual_output, "input_str": input_str}
    else:
        if variant == "output":
            return {"response": FAIL, "correct_output": actual_output, "input_str": input_str}
        else:
            return {"response": FAIL, "correct_output": None, "input_str": input_str}


def evaluate_code_correctness(code: str, hidden_tests: list, timeout: int = 10) -> bool:
    """
    Check if generated code passes ALL hidden tests.

    Uses multiprocessing to properly terminate infinite loops.
    Reads result from Queue BEFORE joining to avoid Windows
    pipe-buffer race conditions.
    """
    q = MPQueue()
    p = Process(target=_evaluate_correctness_worker, args=(code, hidden_tests, q))
    p.start()
    try:
        result = q.get(timeout=timeout)
    except Exception:
        result = False
    finally:
        if p.is_alive():
            p.terminate()
        p.join(timeout=5)
    return result
