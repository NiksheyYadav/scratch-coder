"""MBPP coding tasks as agent tasks: a git workspace with test_solution.py to satisfy.

MBPP (Austin et al., 2021, CC-BY-4.0): 974 short Python problems with assert-based tests.
Splits train/validation/prompt feed agent fine-tuning; test (500) is the held-out eval.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile

from scoder.chat import SUBMIT_COMMAND

from .shell import run

MBPP_REPO = "google-research-datasets/mbpp"

TASK_TEMPLATE = """<pr_description>
Consider the following task:
{text}

Implement it in `solution.py`. The tests in `test_solution.py` must pass:
{tests}
</pr_description>

<instructions>
You are in the root of the repository. Write your solution to `solution.py`, check it with `python test_solution.py`, and fix any errors you see.

When the tests pass, submit your work by issuing exactly the following command:

```bash
{submit}
```

You cannot continue working on this task after submitting.
</instructions>"""


def load_mbpp(split: str) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    from scoder.hub import read_token

    fs = HfFileSystem(token=read_token())
    paths = sorted(fs.glob(f"datasets/{MBPP_REPO}/full/{split}-*.parquet"))
    if not paths:
        raise RuntimeError(f"MBPP split {split} not found")
    rows = []
    for p in paths:
        rows += pq.read_table(fs.open(p, "rb")).to_pylist()
    for r in rows:
        r["code"] = r["code"].replace("\r\n", "\n").replace("\t", "    ")
    return rows


def test_file(problem: dict) -> str:
    setup = problem.get("test_setup_code") or ""
    return "from solution import *\n" + (setup + "\n" if setup.strip() else "") + "\n".join(problem["test_list"]) + \
        "\nprint('ALL TESTS PASSED')\n"


def task_text(problem: dict) -> str:
    return TASK_TEMPLATE.format(text=problem["text"].strip(), tests="\n".join(problem["test_list"]),
                                submit=SUBMIT_COMMAND)


def make_workspace(problem: dict, root: str | None = None) -> str:
    ws = tempfile.mkdtemp(prefix=f"mbpp{problem['task_id']}_", dir=root)
    with open(os.path.join(ws, "test_solution.py"), "w", encoding="utf-8", newline="\n") as f:
        f.write(test_file(problem))
    env = dict(os.environ, GIT_AUTHOR_NAME="agent", GIT_AUTHOR_EMAIL="agent@localhost",
               GIT_COMMITTER_NAME="agent", GIT_COMMITTER_EMAIL="agent@localhost")
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"], ["git", "commit", "-q", "-m", "task"]):
        subprocess.run(cmd, cwd=ws, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return ws


def score(problem: dict, ws: str) -> bool:
    """Re-write the tests (in case the agent edited them) and run them."""
    with open(os.path.join(ws, "test_solution.py"), "w", encoding="utf-8", newline="\n") as f:
        f.write(test_file(problem))
    rc, out = run("python test_solution.py", ws, timeout=30)
    return rc == 0 and "ALL TESTS PASSED" in out


def function_name(problem: dict) -> str | None:
    defs = re.findall(r"^def (\w+)\s*\(", problem["code"], re.M)
    tests = "\n".join(problem["test_list"])
    for d in defs:
        if re.search(rf"\b{re.escape(d)}\s*\(", tests):
            return d
    return defs[-1] if defs else None
