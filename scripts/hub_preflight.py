"""Fail fast unless HF_TOKEN can create and write the checkpoint repo.

    python scripts/hub_preflight.py yourname/scratch-coder
"""
import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoder.hub import describe_hub_error, get_hf_token, token_problem  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print("usage: python scripts/hub_preflight.py yourname/scratch-coder")
        return 2
    repo = sys.argv[1]
    token = get_hf_token()
    if not token:
        print("HF_TOKEN is not set")
        return 1
    problem = token_problem(token)
    if problem:  # catch a wrong/cut-off key before any request or GPU time
        print(f"HF_TOKEN is not usable: {problem}")
        return 1
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        user = api.whoami()["name"]
        api.create_repo(repo, private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=io.BytesIO(f"run started {time.ctime()}\n".encode()),
                        path_in_repo="RUN_STARTED.txt", repo_id=repo, commit_message="pre-flight write check")
    except Exception as e:
        print(f"cannot write to https://huggingface.co/{repo}: {describe_hub_error(e)}")
        return 1
    print(f"pre-flight OK: logged in as {user}, can write to https://huggingface.co/{repo} (private)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
