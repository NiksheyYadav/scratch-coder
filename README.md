# scratch-coder

A small coding agent trained **from scratch**, tokenizer to weights, on a **$1.81** GPU budget.

It reads a task, runs shell commands, reads the output, fixes its mistakes and submits,
using the same protocol as [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent).
Nothing is borrowed from an existing model.

> **Expectations.** $1.81 buys about 10–13 GPU-hours on an RTX 3090, which is enough for a
> ~110M-parameter model on roughly 2B tokens. That is a real model you own end to end, but
> it is small: expect it to solve only easy MBPP-style functions some of the time. The
> pipeline is built so a bigger budget scales by changing one preset and one number.

## What's different from the 2024 nanoGPT recipe

| | 2024 baseline | scratch-coder |
|---|---|---|
| Data | one language (usually Python) | 11 languages from Stack-Edu (Python, JS, TS, Shell, C, C++, Java, Go, Rust, SQL, Markdown docs) **plus 66k real coding-agent sessions** mixed into pretraining |
| End of training | constant mix | **annealing**: during the LR cooldown the mix shifts toward agent sessions, Python and shell |
| Optimizer | AdamW | **Muon** for weight matrices (fewer tokens to the same loss), Adam for embeddings/head |
| Architecture | GPT-2 (LayerNorm, learned positions, GELU) | RMSNorm, rotary positions, **QK-norm**, squared-ReLU MLP, zero-init projections, logit soft-capping |
| Schedule | fixed number of steps | **wall-clock budget**: warmup → constant → cooldown sized to the money you have |
| Data pipeline | tokenize everything first, then train | **streams** parquet row groups from the Hub and tokenizes while the GPU trains |
| After pretraining | — | agent fine-tuning on **executed** trajectories (real tracebacks, real fixes), then an automatic eval that re-runs hidden tests |
| Beyond training | — | a **weight-space programming** toolkit to read, probe and directly edit the trained weights |

## Pipeline

```
tokenizer (trained on our own mix, committed)            tokenizer/tokenizer.json
   |
pretrain  --  streamed mix of 11 languages + agent sessions, Muon, time-boxed     train.py --stage pretrain
   |          last 25% of time: cooldown + anneal mix
   |
agent SFT --  MBPP tasks turned into trajectories by actually running them         agent/make_sft.py
   |          (direct / explore-first / make-a-mistake-and-recover)                  train.py --stage sft
   |
eval      --  held-out MBPP test tasks in a real shell; pass@1 + failure modes     agent/eval.py
   |
weight-space programming -- read the weights, find what causes a failure,          wsp/
              edit it directly, re-evaluate
```

## The paid run on vast.ai

1. **Hugging Face.** Create a *write* token at <https://huggingface.co/settings/tokens>.
   Checkpoints go to a private model repo under your account, so nothing is lost when the credit runs out.
2. **Rent the GPU.** In the vast.ai console, search for a verified, on-demand **RTX 3090**
   (best value at ~$0.14/hr in November 2026; a 4090 or A100 also works). Use a PyTorch template
   (CUDA 12, PyTorch ≥ 2.4) and a **20 GB** disk: data streams, so nothing large is stored.
3. **Start it** (SSH or the Jupyter terminal):

   ```bash
   git clone https://github.com/NiksheyYadav/scratch-coder && cd scratch-coder
   export HF_TOKEN=hf_...                      # your write token
   export HF_REPO=your-hf-username/scratch-coder
   CREDIT=1.81 PRICE_PER_HR=0.135 nohup bash scripts/vast_run.sh > run.log 2>&1 &
   tail -f run.log
   ```

   Set `CREDIT` to your balance and `PRICE_PER_HR` to the price vast.ai shows for the instance.
   The script keeps 10% back for storage and bandwidth and sizes pretraining to the rest.
   If the repo is private, clone it with a GitHub token or copy the folder over with `scp`.
4. **Watch it.** `run.log` prints loss, tokens/s and the data mix. Every 20 minutes a checkpoint
   lands in your Hugging Face repo with `pretrain_metrics.jsonl`, so you can follow it from your phone.
5. **When it prints `ALL DONE`, destroy the instance** in the console so storage stops billing.

## What has been checked (CPU, before spending any credit)

- `scripts/smoke_test.sh` runs every stage end to end on a laptop CPU: streamed pretraining,
  resume, a failing Hub upload that must not stop training, sampling, agent data, fine-tuning,
  eval, and weight inspection, editing and re-evaluation.
- **Muon vs AdamW**, same tiny model, same data, 300 steps: validation loss **5.93 vs 7.01**.
  A tiny-scale sanity check, not a benchmark.
- **Data streaming** from the Hub over a home connection: ~267k tokens/s across all 12 sources,
  about 5× what an RTX 3090 consumes, with the mix matching its target shares.
- **Harness**: replaying known-good trajectories passes and submits 4/4 tasks; qwen2.5:3b through
  Ollama runs end to end, and its format mistakes are caught.
- Not checked: the CUDA path (bf16, `torch.compile`, GPU memory). The paid run falls back to a
  smaller micro-batch on out-of-memory and pushes checkpoints as it goes.

## Run it on a laptop (free, CPU)

```bash
pip install -r requirements.txt
# 3-minute pretraining smoke test on a ~2M-parameter model
python train.py --stage pretrain --preset nano --mix smoke --time_min 3 --out out/smoke/pretrain
python sample.py --ckpt out/smoke/pretrain/ckpt.pt --prompt "def add(a, b):"
python -m agent.make_sft --out out/smoke/sft.jsonl --limit 20
python train.py --stage sft --init_from out/smoke/pretrain/ckpt.pt --sft_data out/smoke/sft.jsonl --time_min 1 --out out/smoke/sft
python -m agent.eval --backend local --ckpt out/smoke/sft/ckpt.pt --limit 3 --out out/smoke/eval
# check the harness with any Ollama model
python -m agent.eval --backend ollama --model qwen2.5:3b --limit 5 --out out/eval_qwen
```

Download the trained model from your Hugging Face repo (`sft.pt`) and use it the same way.
On Windows the harness uses Git Bash (`C:\Program Files\Git\bin\bash.exe`), not WSL.

**Safety:** the agent runs real shell commands inside a temporary task folder. A small
blocklist stops obviously destructive commands, but it is not a sandbox. Run large
evaluations inside a container or VM; a vast.ai instance is one.

## Weight-space programming

Inspired by MIT CSAIL's *Sparks of Weight-Space Programming* (Nov 2026), where a frontier
LLM removed a "loop direction" from Qwen3.5-0.8B's MLPs by editing 0.016% of its weights,
raising MATH-500 from 7.9% to 48.1%. Their own finding matters here: **editing removed an
obstacle that hid an ability; it could not add an ability the model lacked.** So the
useful targets are failure modes like runaway generation, format errors and repeated
commands, which `agent.eval` measures.

```bash
# 1. evaluate and log every reply
python -m agent.eval --backend local --ckpt out/sft/ckpt.pt --limit 100 --out out/eval
# 2. read the weights and probe bad vs good replies; saves per-layer directions
python -m wsp.inspect_weights --ckpt out/sft/ckpt.pt --probe out/eval/episodes.jsonl --out out/wsp/report.json
python -m wsp.inspect_weights --ckpt out/sft/ckpt.pt --dump blocks.11.mlp.proj.weight --rows 0:8
# 3. write edits.json (by hand, or ask an LLM to read report.json and propose edits), apply them
python -m wsp.edit --ckpt out/sft/ckpt.pt --edits edits.json --out out/wsp/edited.pt
# 4. re-evaluate the edited model on the same tasks and keep the edit only if it helps
python -m agent.eval --backend local --ckpt out/wsp/edited.pt --limit 100 --out out/eval_edited
```

Edit operations: `scale` rows/columns (e.g. silence MLP neurons), `set` single weights,
`ablate_head`, `project_out` a residual direction from what a matrix writes (the MIT loop
fix), and `add_direction`. Every edited checkpoint records its edits and how many weights changed.

## Layout

```
scoder/config.py     model presets (nano 2M ... base 110M) and the data mixes
scoder/data.py       streaming multi-source loader, validation sets, SFT batches
scoder/model.py      the transformer, KV-cached generation
scoder/optim.py      Muon + Adam setup
scoder/tokenizer.py  BPE tokenizer training (16k vocab, code-aware split)
scoder/chat.py       chat template and the agent protocol
scoder/hub.py        background, non-fatal checkpoint uploads
train.py             pretraining and agent fine-tuning
agent/               shell runner, MBPP tasks, trajectory builder, harness, eval
wsp/                 weight inspection, probing and editing
scripts/vast_run.sh  the whole paid run in one command
data/stack_edu_manifest.json  which Stack-Edu parquet file holds which language
```

## Data and licences

- **Stack-Edu** ([HuggingFaceTB/stack-edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu),
  streamed from the text mirror `utter-project/stack-edu-HF`): educational code filtered from The Stack v2.
  The Stack's terms of use apply, including respecting the original file licences.
- **Agent sessions**: [Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k](https://huggingface.co/datasets/Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k) (MIT).
- **MBPP** ([google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp), CC-BY-4.0):
  train/validation/prompt splits for agent fine-tuning, test split for evaluation.

## Next stages

- **RL with test rewards (GRPO):** sample several agent attempts per task, reward passing tests.
  Worth it once the base model passes a few percent of tasks.
- **More agent data** in the anneal mix, and a longer context for long sessions.
- **A bigger budget:** `--preset base` → wider/deeper preset, same scripts.
