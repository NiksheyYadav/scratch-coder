"""The coding agent loop: think, run one shell command, read the output, repeat, submit.

Backends: our own checkpoint ("local") or any Ollama model ("ollama", handy for
checking the harness itself with a capable model).
"""
from __future__ import annotations

import json
import urllib.request

import torch

from scoder.chat import END, EOT, FINAL_MARKER, FORMAT_ERROR, SYSTEM_PROMPT, format_observation, parse_actions, render

from . import shell


class LocalBackend:
    def __init__(self, ckpt: str, device: str | None = None, temperature: float = 0.2, max_new_tokens: int = 512,
                 repetition_penalty: float = 1.0, seed: int = 0):
        from scoder.model import load_checkpoint
        from scoder.tokenizer import load_tokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _ = load_checkpoint(ckpt, self.device)
        self.tok = load_tokenizer()
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.stops = {self.tok.token_to_id(END), self.tok.token_to_id(EOT)}
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.name = f"local:{ckpt}"

    def _fit(self, messages: list[dict]) -> list[int]:
        """Encode, dropping the oldest turns after the task if the context would overflow."""
        limit = self.model.cfg.seq_len - self.max_new_tokens
        msgs = list(messages)
        while True:
            ids = self.tok.encode(render(msgs, add_generation_prompt=True)).ids
            if len(ids) <= limit or len(msgs) <= 3:
                return ids[-limit:]
            del msgs[2:4]  # oldest assistant turn and its observation

    def complete(self, messages: list[dict]) -> tuple[str, bool]:
        ids = self._fit(messages)
        out = self.model.generate(ids, self.max_new_tokens, stop_ids=self.stops, temperature=self.temperature,
                                  repetition_penalty=self.repetition_penalty, generator=self.gen)
        hit_limit = not out or out[-1] not in self.stops
        if out and out[-1] in self.stops:
            out = out[:-1]
        return self.tok.decode(out, skip_special_tokens=True), hit_limit


class OllamaBackend:
    def __init__(self, model: str, url: str = "http://localhost:11434", temperature: float = 0.2,
                 max_new_tokens: int = 512):
        self.model, self.url = model, url.rstrip("/")
        self.options = {"temperature": temperature, "num_predict": max_new_tokens, "num_ctx": 8192}
        self.max_new_tokens = max_new_tokens
        self.name = f"ollama:{model}"

    def complete(self, messages: list[dict]) -> tuple[str, bool]:
        body = json.dumps({"model": self.model, "messages": messages, "stream": False, "options": self.options}).encode()
        req = urllib.request.Request(f"{self.url}/api/chat", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            res = json.loads(r.read())
        return res["message"]["content"], res.get("done_reason") == "length"


def run_episode(backend, task: str, workspace: str, max_steps: int = 8, timeout: int = 20) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task}]
    stats = {"steps": 0, "format_errors": 0, "repeated_actions": 0, "hit_token_limit": 0, "submitted": False,
             "stop_reason": "max_steps"}
    seen: set[str] = set()
    for _ in range(max_steps):
        stats["steps"] += 1
        reply, hit_limit = backend.complete(messages)
        stats["hit_token_limit"] += int(hit_limit)
        messages.append({"role": "assistant", "content": reply})
        actions = parse_actions(reply)
        if len(actions) != 1:
            stats["format_errors"] += 1
            messages.append({"role": "user", "content": FORMAT_ERROR.format(n=len(actions))})
            continue
        cmd = actions[0]
        if cmd in seen:
            stats["repeated_actions"] += 1
        seen.add(cmd)
        rc, out = shell.run(cmd, workspace, timeout=timeout)
        if out.lstrip().startswith(FINAL_MARKER):
            stats["submitted"] = True
            stats["stop_reason"] = "submitted"
            break
        messages.append({"role": "user", "content": format_observation(rc, out)})
    stats["messages"] = messages
    return stats
