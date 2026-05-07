"""
utils/prompts.py
Prompt templates for code and test generation.
Matches the exact prompts from the TICODER paper's GitHub repo
(https://github.com/microsoft/ticoder/blob/main/src/query_chat_model.py).
"""


def build_code_prompt(description: str, func_header: str) -> dict:
    """
    Build the code generation prompt matching the paper's repo exactly.

    Returns:
        {"system": system_message, "user": user_message}
    """
    system = (
        "Suppose you are a code completion engine.\n"
        "You are asked to complete the following Python function. "
        "The function signature is given below. The context of the function "
        "is also provided. Complete the function.\n"
    )

    user = (
        f"Complete the following Python function:\n\n"
        f"{func_header}\n\n"
        f"The context of the function is :\n\n"
        f"{description}\n\n"
        f"Surround the function with ` and ` tags.\n"
        f"Do not explain the function, just complete the function.\n"
    )

    return {"system": system, "user": user}


def build_test_prompt(description: str, func_header: str, func_name: str) -> dict:
    """
    Build the test generation prompt matching the paper's repo exactly.

    Returns:
        {"system": system_message, "user": user_message}
    """
    system = (
        "Suppose you are a code completion engine. "
        "You are asked to generate tests for a Python function. \n"
        "You will be given a function which contains the description. \n"
        "You need to generate tests for the function. "
    )

    code_stub = f"{func_header}\n    \"\"\"{description}\"\"\"\n    pass"

    user = (
        f"Context of the function is :\n\n"
        f"{description}\n\n"
        f"The functions is defined as follows:\n\n"
        f"{code_stub}\n\n"
        f"Generate a test code for the function containing assersions. \n"
        f"Start the test code with: \n\n"
        f"def test_{func_name}():\n"
        f"\tassert {func_name} (\n\n\n"
        f"Surround the test code with ` and ` tags.\n"
        f"Do not explain the test code, just generate it. "
        f"Do not call the test code.\n"
        f"Do not write any standalone asserts.\n"
        f"The test code should contain only one assertion for the function. \n"
    )

    return {"system": system, "user": user}


def extract_func_name(func_header: str) -> str:
    """Extract function name from header like 'def my_func(args):'"""
    try:
        name = func_header.strip().split("def ")[1].split("(")[0].strip()
        return name
    except Exception:
        return "func"


def extract_func_header(code: str) -> str:
    """Extract the function header (first line) from full code."""
    for line in code.strip().split("\n"):
        if line.strip().startswith("def "):
            return line.strip()
    return ""
