"""Background checkpoint uploads to the Hugging Face Hub. Failures are logged, never fatal."""
from __future__ import annotations

import os
import time

import re

_BEARER = re.compile(r"^bearer\s+", re.IGNORECASE)
# Key prefixes of other services, to give a precise hint when the wrong key is pasted.
_OTHER_SERVICES = {
    "cnx_": "CodeNex", "nvapi-": "NVIDIA", "sk-": "OpenAI/Anthropic", "ghp_": "GitHub", "github_pat_": "GitHub",
}
TOKEN_URL = "https://huggingface.co/settings/tokens"


def clean_token(raw: str | None) -> str | None:
    """Undo common copy/paste damage: whitespace, newlines/CRLF, a BOM, quotes, a 'Bearer ' prefix."""
    if raw is None:
        return None
    s = raw.replace("\ufeff", "").strip().strip("\"'").strip()
    s = _BEARER.sub("", s)
    s = re.sub(r"\s+", "", s).strip("\"'")
    return s or None


def token_problem(token: str) -> str | None:
    """Why this value cannot be a Hugging Face token, or None. Never echoes the value."""
    for prefix, service in _OTHER_SERVICES.items():
        if token.startswith(prefix):
            return (f"This looks like a {service} key, not a Hugging Face token. "
                    f"Hugging Face tokens start with 'hf_' (create one at {TOKEN_URL}).")
    if not token.startswith("hf_"):
        return f"A Hugging Face token starts with 'hf_'; this value does not. Check you copied all of it ({TOKEN_URL})."
    if len(token) < 20:
        return "This Hugging Face token is too short; it was probably cut off when copying."
    return None


def get_hf_token() -> str | None:
    """HF_TOKEN (or the legacy HUGGING_FACE_HUB_TOKEN), cleaned. None when unset."""
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        tok = clean_token(os.environ.get(name))
        if tok:
            return tok
    return None


def read_token() -> str | None:
    """Token for READING public data. A malformed token is ignored (with a warning) instead of
    making Hugging Face answer 401 to requests that would have worked anonymously."""
    tok = get_hf_token()
    problem = token_problem(tok) if tok else None
    if problem:
        print(f"[hub] WARNING: ignoring HF_TOKEN for downloads: {problem}", flush=True)
        return None
    return tok


def describe_hub_error(e: Exception) -> str:
    """Explain a Hub failure in plain words, telling a bad token (401) from a missing permission (403)."""
    status = getattr(getattr(e, "response", None), "status_code", None)
    if status == 401:
        return ("Hugging Face said the token was not accepted (401): it is wrong, expired, revoked, or has "
                f"stray characters. Create a new token at {TOKEN_URL}.")
    if status == 403:
        return ("The token is valid but has no WRITE permission for that repo or namespace (403). "
                "Use a token with write access, or fix HF_REPO.")
    if status == 404:
        return "Repo or namespace not found (404). Check HF_REPO is yourname/repo and that the account exists."
    return repr(e)



class HubPusher:
    def __init__(self, repo_id: str | None, private: bool = True):
        self.repo_id = repo_id
        self.future = None
        self.api = None
        self.failures = 0
        if not repo_id:
            return
        try:
            from huggingface_hub import HfApi

            self.api = HfApi(token=get_hf_token())
            self.api.create_repo(repo_id, private=private, exist_ok=True)
            print(f"[hub] pushing checkpoints to https://huggingface.co/{repo_id} (private={private})", flush=True)
        except Exception as e:
            print(f"[hub] WARNING: cannot use {repo_id}: {describe_hub_error(e)} Training continues with local checkpoints only.", flush=True)
            self.api = None

    def push(self, folder: str, message: str):
        if self.api is None:
            return
        if self.future is not None and not self.future.done():
            print("[hub] previous upload still running; skipping this one", flush=True)
            return
        try:
            if self.future is not None and self.future.exception() is not None:
                self.failures += 1
                print(f"[hub] WARNING: previous upload failed: {self.future.exception()!r}", flush=True)
            self.future = self.api.upload_folder(repo_id=self.repo_id, folder_path=folder, ignore_patterns=["*.tmp"],
                                                 commit_message=message, run_as_future=True)
        except Exception as e:
            self.failures += 1
            print(f"[hub] WARNING: upload failed to start: {e!r}", flush=True)

    def wait(self, timeout: float = 900):
        if self.future is None:
            return
        t = time.time()
        while not self.future.done() and time.time() - t < timeout:
            time.sleep(2)
        if self.future.done() and self.future.exception() is None:
            print("[hub] final upload complete", flush=True)
        else:
            print(f"[hub] WARNING: final upload not confirmed: {self.future.exception() if self.future.done() else 'timeout'}", flush=True)
