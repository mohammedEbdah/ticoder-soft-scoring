"""
data/download_datasets.py
Downloads and prepares the MBPP dataset for TICODER experiments.
"""

import json
import os
import sys
import urllib.request

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.safe_io import atomic_write_jsonl

# MBPP sanitized dataset (same version used in paper)
MBPP_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json"
MBPP_OUTPUT = "data/mbpp.jsonl"


def download_mbpp():
    """Download and parse the MBPP sanitized dataset."""
    os.makedirs("data", exist_ok=True)

    print("Downloading MBPP dataset...")
    try:
        urllib.request.urlretrieve(MBPP_URL, "data/mbpp_raw.json")
        print("Downloaded successfully.")
    except Exception as e:
        print(f"Download failed: {e}")
        print("Please manually download MBPP from:")
        print("https://github.com/google-research/google-research/tree/master/mbpp")
        return

    # Parse and save as JSONL
    with open("data/mbpp_raw.json", "r") as f:
        data = json.load(f)

    examples = []
    for item in data:
        # Sanitized MBPP uses "prompt" for description; normalize to "text"
        description = item.get("prompt", item.get("text", ""))
        example = {
            "task_id": item.get("task_id"),
            "text": description,                     # Natural language description
            "code": item.get("code", ""),             # Reference (oracle) implementation
            "test_list": item.get("test_list", []),   # Hidden test cases
            "test_setup_code": item.get("test_setup_code", ""),
        }
        examples.append(example)

    atomic_write_jsonl(MBPP_OUTPUT, examples)

    print(f"Saved {len(examples)} examples to {MBPP_OUTPUT}")
    return examples


def load_mbpp(path=MBPP_OUTPUT, limit=None):
    """Load MBPP dataset from JSONL file."""
    examples = []
    with open(path, "r") as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    if limit:
        examples = examples[:limit]
    print(f"Loaded {len(examples)} MBPP examples.")
    return examples


if __name__ == "__main__":
    download_mbpp()
