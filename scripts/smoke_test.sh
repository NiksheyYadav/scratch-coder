#!/usr/bin/env bash
# CPU smoke test of every stage the paid run uses (about 10 minutes, downloads ~300 MB).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
O=out/smoke
rm -rf "$O" && mkdir -p "$O"

echo "== pretrain (nano, 2 min, streamed data, failing Hub push must not stop training)"
python train.py --stage pretrain --preset nano --mix smoke --time_min 2 --out $O/pretrain \
  --log_every 5 --eval_every_min 1 --ckpt_every_min 1 --hf_repo no-such-user-0/scratch-coder-smoke
echo "== resume"
python train.py --stage pretrain --preset nano --mix smoke --time_min 2.5 --resume --out $O/pretrain --log_every 5
echo "== sample"
python sample.py --ckpt $O/pretrain/ckpt.pt --prompt "def add(a, b):" --max_new_tokens 30
echo "== agent data"
python -m agent.make_sft --out $O/sft.jsonl --limit 5
echo "== agent fine-tune"
python train.py --stage sft --init_from $O/pretrain/ckpt.pt --sft_data $O/sft.jsonl --time_min 0.5 --out $O/sft --log_every 10
echo "== eval"
python -m agent.eval --backend local --ckpt $O/sft/ckpt.pt --limit 2 --max_steps 2 --max_new_tokens 32 --out $O/eval
echo "== weight-space programming"
python -m wsp.inspect_weights --ckpt $O/sft/ckpt.pt --probe $O/eval/episodes.jsonl --out $O/wsp/report.json
cat > $O/wsp/edits.json <<'EOF'
[{"op": "ablate_head", "layer": 0, "head": 1}, {"op": "scale", "param": "blocks.1.mlp.proj.weight", "cols": [0, 1], "factor": 0.5}]
EOF
python -m wsp.edit --ckpt $O/sft/ckpt.pt --edits $O/wsp/edits.json --out $O/wsp/edited.pt
python -m agent.eval --backend local --ckpt $O/wsp/edited.pt --limit 1 --max_steps 1 --max_new_tokens 16 --out $O/eval_edited
echo "SMOKE TEST PASSED"
