"""Score an agent on held-out MBPP tasks, with the diagnostics weight editing needs.

    python -m agent.eval --backend local --ckpt out/sft/ckpt.pt --limit 100 --out out/eval
    python -m agent.eval --backend ollama --model qwen2.5:3b --limit 10 --out out/eval_qwen

Besides pass@1 it reports how the agent fails: format errors (no single bash block),
replies that ran into the token limit (runaway/looping generation), repeated commands
and never submitting. Those are the failure modes a weight edit can target.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

from .harness import LocalBackend, OllamaBackend, run_episode
from .tasks import load_mbpp, make_workspace, score, task_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["local", "ollama"], default="local")
    ap.add_argument("--ckpt", help="checkpoint for --backend local")
    ap.add_argument("--model", default="qwen2.5:3b", help="model for --backend ollama")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--out", default="out/eval")
    args = ap.parse_args()

    if args.backend == "local":
        backend = LocalBackend(args.ckpt, temperature=args.temperature, max_new_tokens=args.max_new_tokens,
                               repetition_penalty=args.repetition_penalty)
    else:
        backend = OllamaBackend(args.model, temperature=args.temperature, max_new_tokens=args.max_new_tokens)
    problems = load_mbpp(args.split)[args.offset : args.offset + args.limit]
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "episodes.jsonl")
    agg = {"tasks": 0, "passed": 0, "submitted": 0, "steps": 0, "format_errors": 0, "replies": 0,
           "hit_token_limit": 0, "repeated_actions": 0}
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        for i, p in enumerate(problems):
            ws = make_workspace(p)
            try:
                ep = run_episode(backend, task_text(p), ws, max_steps=args.max_steps)
                passed = score(p, ws)
            finally:
                shutil.rmtree(ws, ignore_errors=True)
            agg["tasks"] += 1
            agg["passed"] += int(passed)
            agg["submitted"] += int(ep["submitted"])
            agg["steps"] += ep["steps"]
            agg["replies"] += ep["steps"]
            agg["format_errors"] += ep["format_errors"]
            agg["hit_token_limit"] += ep["hit_token_limit"]
            agg["repeated_actions"] += ep["repeated_actions"]
            log.write(json.dumps({"task_id": p["task_id"], "passed": passed, **ep}) + "\n")
            print(f"[{i + 1}/{len(problems)}] task {p['task_id']}: {'PASS' if passed else 'fail'} "
                  f"steps={ep['steps']} submitted={ep['submitted']} format_err={ep['format_errors']} "
                  f"runaway={ep['hit_token_limit']}", flush=True)
    n, r = max(1, agg["tasks"]), max(1, agg["replies"])
    summary = {
        "backend": backend.name, "split": args.split, "tasks": agg["tasks"],
        "pass@1": round(agg["passed"] / n, 4),
        "submit_rate": round(agg["submitted"] / n, 4),
        "avg_steps": round(agg["steps"] / n, 2),
        "format_error_rate": round(agg["format_errors"] / r, 4),
        "runaway_reply_rate": round(agg["hit_token_limit"] / r, 4),
        "repeated_action_rate": round(agg["repeated_actions"] / r, 4),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
