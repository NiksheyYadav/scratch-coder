"""Build agent fine-tuning trajectories from MBPP by actually running them.

Every observation in the data is the real output of the command in a real workspace,
so the model learns what tracebacks, empty outputs and passing tests look like.
Three shapes of trajectory:
  direct   - write solution -> run tests -> submit
  explore  - cat the tests -> write -> run tests -> submit
  recover  - write a broken version -> read the error -> fix -> run tests -> submit

    python -m agent.make_sft --out data/sft_mbpp.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil

from scoder.chat import SUBMIT_COMMAND, SYSTEM_PROMPT, format_observation

from .shell import run
from .tasks import function_name, load_mbpp, make_workspace, task_text

WRITE_INTROS = [
    "The tests call `{fn}`. I'll implement it in solution.py.",
    "I need a function `{fn}` that satisfies the tests. Let me write it to solution.py.",
    "From the task and the asserts, `{fn}` should do the following: {gist} I'll write the implementation now.",
    "Let me create solution.py with an implementation of `{fn}`.",
]
TEST_INTROS = [
    "Now let me run the tests to check the solution.",
    "Let me verify the implementation by running the tests.",
    "I'll run the test file to make sure everything passes.",
]
SUBMIT_INTROS = [
    "All tests pass. I'll submit the solution.",
    "The tests pass, so the task is done. Submitting.",
    "Everything passes. Time to submit.",
]
EXPLORE_INTROS = [
    "Let me look at the test file first to see exactly how the function is called.",
    "First, I'll read test_solution.py to understand the expected interface.",
]


def sanitize(text: str, ws: str) -> str:
    for form in {ws, ws.replace("\\", "/"), os.path.realpath(ws)}:
        text = text.replace(form, "/testbed")
    # any remaining Windows path (e.g. the Python install) -> a neutral Linux-looking path
    text = re.sub(r"[A-Za-z]:[\\/][^\s\"']*[\\/]", "/usr/lib/python3/", text)
    return text.replace("\\", "/") if "/testbed" in text else text


def write_cmd(code: str) -> str:
    return f"cat <<'EOF' > solution.py\n{code.rstrip()}\nEOF"


def thought(msg: str, cmd: str) -> str:
    return f"THOUGHT: {msg}\n\n```bash\n{cmd}\n```"


def break_code(code: str, fn: str, rng: random.Random) -> tuple[str, str] | None:
    """Return (broken_code, explanation) for a realistic first mistake."""
    kinds = ["rename", "colon"]
    rng.shuffle(kinds)
    for k in kinds:
        if k == "rename" and re.search(rf"^def {re.escape(fn)}\s*\(", code, re.M):
            wrong = fn + rng.choice(["_func", "_impl", "_solution"])
            return (re.sub(rf"^def {re.escape(fn)}(\s*\()", rf"def {wrong}\1", code, count=1, flags=re.M),
                    f"The tests cannot find `{fn}`: I named the function `{wrong}` by mistake. Let me fix the name.")
        if k == "colon":
            m = re.search(rf"^(def {re.escape(fn)}\s*\(.*\))\s*:", code, re.M)
            if m:
                return (code[: m.start()] + m.group(1) + code[m.end():],
                        "There is a SyntaxError: the `def` line is missing its colon. Let me rewrite the file correctly.")
    return None


def build_one(problem: dict, shape: str, rng: random.Random, root: str | None) -> dict | None:
    fn = function_name(problem)
    code = problem["code"].strip()
    if not fn or "\nEOF" in code:
        return None
    ws = make_workspace(problem, root)
    try:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task_text(problem)}]

        def step(msg: str, cmd: str) -> tuple[int, str]:
            msgs.append({"role": "assistant", "content": thought(msg, cmd)})
            rc, out = run(cmd, ws, timeout=30)
            msgs.append({"role": "user", "content": format_observation(rc, sanitize(out, ws))})
            return rc, out

        gist = problem["text"].strip().rstrip(".") + "."
        if shape == "explore":
            step(rng.choice(EXPLORE_INTROS), "cat test_solution.py")
        if shape == "recover":
            broken = break_code(code, fn, rng)
            if broken is None:
                return None
            bad, why = broken
            step(rng.choice(WRITE_INTROS).format(fn=fn, gist=gist), write_cmd(bad))
            rc, out = step(rng.choice(TEST_INTROS), "python test_solution.py")
            if rc == 0:
                return None  # the "mistake" did not break anything; skip
            step(why, write_cmd(code))
        else:
            step(rng.choice(WRITE_INTROS).format(fn=fn, gist=gist), write_cmd(code))
        rc, out = step(rng.choice(TEST_INTROS), "python test_solution.py")
        if rc != 0 or "ALL TESTS PASSED" not in out:
            return None  # reference solution does not pass here; drop it
        msgs.append({"role": "assistant", "content": thought(rng.choice(SUBMIT_INTROS), SUBMIT_COMMAND)})
        return {"task_id": problem["task_id"], "shape": shape, "messages": msgs}
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_mbpp.jsonl")
    ap.add_argument("--splits", default="train,validation,prompt")
    ap.add_argument("--variants", type=int, default=3, help="trajectories per problem")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    problems = []
    for s in args.splits.split(","):
        problems += load_mbpp(s)
    if args.limit:
        problems = problems[: args.limit]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    counts = {"direct": 0, "explore": 0, "recover": 0, "dropped": 0}
    with open(args.out, "w", encoding="utf-8") as f:
        for i, p in enumerate(problems):
            shapes = ["direct", "explore", "recover"][: args.variants]
            for shape in shapes:
                ex = build_one(p, shape, rng, None)
                if ex is None:
                    counts["dropped"] += 1
                    continue
                counts[shape] += 1
                f.write(json.dumps(ex) + "\n")
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(problems)} problems: {counts}", flush=True)
    print(f"wrote {args.out}: {counts}")


if __name__ == "__main__":
    main()
