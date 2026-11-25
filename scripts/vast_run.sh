#!/usr/bin/env bash
# One command for the whole paid run on a vast.ai GPU:
#   pretrain (time-boxed to your credit) -> build agent data -> agent fine-tune -> quick eval
# Every checkpoint is pushed to your private Hugging Face repo as it goes, so nothing is
# lost when the credit runs out and the instance stops.
#
# Usage (inside the instance, from the repo root):
#   export HF_TOKEN=hf_...                     # a *write* token from huggingface.co/settings/tokens
#   export HF_REPO=your-hf-username/scratch-coder
#   CREDIT=1.81 PRICE_PER_HR=0.135 nohup bash scripts/vast_run.sh > run.log 2>&1 &
#   tail -f run.log
set -euo pipefail
cd "$(dirname "$0")/.."

: "${HF_TOKEN:?set HF_TOKEN to a Hugging Face write token}"
: "${HF_REPO:?set HF_REPO, e.g. yourname/scratch-coder}"
: "${CREDIT:?set CREDIT to your remaining vast.ai balance in dollars, e.g. 1.81}"
: "${PRICE_PER_HR:?set PRICE_PER_HR to the instance price shown by vast.ai, e.g. 0.135}"
PRESET=${PRESET:-base}
SFT_MIN=${SFT_MIN:-8}    # upper bound; fine-tuning stops after 3 passes over the data
EVAL_TASKS=${EVAL_TASKS:-50}
SAFETY=${SAFETY:-0.90}   # keep 10% of the credit for storage, bandwidth and slack

TOTAL_MIN=$(python -c "print(int($CREDIT / $PRICE_PER_HR * 60 * $SAFETY))")
OVERHEAD_MIN=${OVERHEAD_MIN:-30}  # setup + eval + final uploads (agent data builds during pretraining)
PRETRAIN_MIN=$((TOTAL_MIN - OVERHEAD_MIN - SFT_MIN))
echo "budget: ${TOTAL_MIN} min total -> pretrain ${PRETRAIN_MIN} min, agent fine-tune ${SFT_MIN} min"
if [ "$PRETRAIN_MIN" -lt 20 ]; then echo "not enough credit for a useful run"; exit 1; fi

export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git)
python -m pip install -q -r requirements.txt

# Pre-flight: the Hugging Face repo is the only place the work survives once the credit
# runs out, so prove we can write to it before spending GPU time.
python scripts/hub_preflight.py "$HF_REPO" || { echo "PRE-FLIGHT FAILED: fix HF_TOKEN / HF_REPO and rerun"; exit 1; }
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "no CUDA GPU visible"
p = torch.cuda.get_device_properties(0)
print(f"GPU: {p.name}, {p.total_memory / 2**30:.0f} GB, torch {torch.__version__}, bf16={torch.cuda.is_bf16_supported()}")
EOF

# 1) agent fine-tuning data from MBPP, built by really running every command.
#    CPU-only, so it runs in the background while the GPU pretrains.
python -m agent.make_sft --out data/sft_mbpp.jsonl > make_sft.log 2>&1 &
SFT_PID=$!

# 2) pretraining from scratch (streams the data mix from the Hub while it trains)
python train.py --stage pretrain --preset "$PRESET" --time_min "$PRETRAIN_MIN" --out out/pretrain --hf_repo "$HF_REPO"

wait "$SFT_PID" || { echo "agent data build failed:"; tail -20 make_sft.log; exit 1; }
tail -1 make_sft.log

# 3) agent fine-tuning
python train.py --stage sft --init_from out/pretrain/ckpt.pt --sft_data data/sft_mbpp.jsonl \
  --time_min "$SFT_MIN" --out out/sft --hf_repo "$HF_REPO"

# 4) quick held-out evaluation (run the full 500 later on any machine)
python -m agent.eval --backend local --ckpt out/sft/ckpt.pt --limit "$EVAL_TASKS" --max_new_tokens 384 --out out/eval || true
python - <<EOF || true
import os
from huggingface_hub import HfApi
HfApi(token=os.environ["HF_TOKEN"]).upload_folder(repo_id="$HF_REPO", folder_path="out/eval", path_in_repo="eval")
print("uploaded eval results")
EOF

echo "ALL DONE. Weights are in https://huggingface.co/$HF_REPO"
echo "Destroy the instance in the vast.ai console now so storage stops billing."
