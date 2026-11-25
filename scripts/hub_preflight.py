"""Fail fast unless HF_TOKEN can create and write the checkpoint repo.

    python scripts/hub_preflight.py yourname/scratch-coder
"""
import io
import os
import sys
import time


def main():
    repo = sys.argv[1]
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set")
        return 1
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        user = api.whoami()["name"]
        api.create_repo(repo, private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=io.BytesIO(f"run started {time.ctime()}\n".encode()),
                        path_in_repo="RUN_STARTED.txt", repo_id=repo, commit_message="pre-flight write check")
    except Exception as e:
        print(f"cannot write to https://huggingface.co/{repo}: {e!r}")
        print("Use a token with WRITE access, and a repo under your own account or an org you can write to.")
        return 1
    print(f"pre-flight OK: logged in as {user}, can write to https://huggingface.co/{repo} (private)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
