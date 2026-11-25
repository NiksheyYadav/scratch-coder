"""Run agent commands in a task workspace with a timeout.

The model's commands run as real shell commands on this machine, inside the task's
temporary folder. Run evaluations in a container or VM (a vast.ai instance is one).
The blocklist below only stops the most obviously destructive commands.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess

_BLOCK = re.compile(
    r"(\brm\s+-[a-zA-Z]*[rf][a-zA-Z]*\s+(/|~|\$HOME|\.\.|\*)(\s|$))|\bmkfs\b|\bdd\s+if=|\bshutdown\b|\breboot\b|:\(\)\s*\{|\bsudo\b"
)


def find_bash() -> str:
    if os.name == "nt":
        candidates = [r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files (x86)\Git\bin\bash.exe"]
        git = shutil.which("git")
        if git:
            candidates.append(os.path.join(os.path.dirname(os.path.dirname(git)), "bin", "bash.exe"))
        for c in candidates:
            if os.path.exists(c):
                return c
        raise RuntimeError("Git Bash not found (the 'bash' on PATH is usually WSL). Install Git for Windows.")
    return shutil.which("bash") or "/bin/bash"


BASH = find_bash()


def run(cmd: str, cwd: str, timeout: int = 20) -> tuple[int, str]:
    if _BLOCK.search(cmd):
        return 1, "Command blocked by the harness safety filter."
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1", GIT_PAGER="cat", PAGER="cat",
               GIT_AUTHOR_NAME="agent", GIT_AUTHOR_EMAIL="agent@localhost",
               GIT_COMMITTER_NAME="agent", GIT_COMMITTER_EMAIL="agent@localhost")
    try:
        p = subprocess.run([BASH, "-c", cmd], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=timeout, env=env)
        return p.returncode, p.stdout.decode("utf-8", errors="replace").replace("\r\n", "\n")
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode("utf-8", errors="replace")
        return -1, out + f"\nCommand timed out after {timeout} seconds."
