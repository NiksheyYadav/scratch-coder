"""Background checkpoint uploads to the Hugging Face Hub. Failures are logged, never fatal."""
from __future__ import annotations

import os
import time


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

            self.api = HfApi(token=os.environ.get("HF_TOKEN"))
            self.api.create_repo(repo_id, private=private, exist_ok=True)
            print(f"[hub] pushing checkpoints to https://huggingface.co/{repo_id} (private={private})", flush=True)
        except Exception as e:
            print(f"[hub] WARNING: cannot use {repo_id}: {e!r}. Training continues with local checkpoints only.", flush=True)
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
            self.future = self.api.upload_folder(repo_id=self.repo_id, folder_path=folder,
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
