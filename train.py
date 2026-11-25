"""Train scratch-coder.

Pretraining (time-boxed, streams data from the Hub, anneals onto agent data at the end):
    python train.py --stage pretrain --preset base --time_min 540 --out out/pretrain --hf_repo you/scratch-coder

Agent fine-tuning on executed MBPP trajectories:
    python train.py --stage sft --init_from out/pretrain/ckpt.pt --sft_data data/sft_mbpp.jsonl --time_min 10 --out out/sft
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import time

import torch

from scoder.config import COOLDOWN_FRAC, MIXES, PRESETS, ModelConfig
from scoder.hub import HubPusher
from scoder.model import GPT
from scoder.optim import build_optimizers, set_lr
from scoder.tokenizer import DEFAULT_PATH as TOKENIZER_PATH, load_tokenizer


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pretrain", "sft"], default="pretrain")
    ap.add_argument("--preset", default="base", choices=list(PRESETS))
    ap.add_argument("--mix", default="pretrain", help="pretrain (switches to anneal in cooldown) or any key of MIXES")
    ap.add_argument("--time_min", type=float, required=True, help="training time budget in minutes")
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--batch_tokens", type=int, default=0, help="tokens per optimizer step (0 = preset default)")
    ap.add_argument("--micro_bs", type=int, default=0, help="sequences per micro-batch (0 = pick from GPU memory)")
    ap.add_argument("--optimizer", choices=["muon", "adamw"], default="muon")
    ap.add_argument("--lr_scale", type=float, default=1.0)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--cooldown_frac", type=float, default=None)
    ap.add_argument("--eval_every_min", type=float, default=15)
    ap.add_argument("--ckpt_every_min", type=float, default=20)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf_repo", default=os.environ.get("HF_REPO") or None)
    ap.add_argument("--init_from", default=None, help="checkpoint to start from (sft)")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/ckpt.pt")
    ap.add_argument("--sft_data", default="data/sft_mbpp.jsonl")
    ap.add_argument("--sft_epochs", type=float, default=3.0)
    ap.add_argument("--compile", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1234)
    return ap.parse_args()


def pick_micro_bs(device: str, cfg: ModelConfig) -> int:
    if device != "cuda":
        return 4
    gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    # calibrated for the 110M "base" preset at 2048 tokens; smaller presets get the same (safe)
    if gb >= 70:
        return 64
    if gb >= 38:
        return 32
    if gb >= 22:
        return 16
    if gb >= 15:
        return 8
    return 4


def lr_mult(step: int, progress: float, warmup: int, cooldown: float) -> float:
    if step < warmup:
        return (step + 1) / warmup
    if progress < 1 - cooldown:
        return 1.0
    return max(0.0, (1 - progress) / cooldown)


def main():
    args = parse()
    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)
    tok = load_tokenizer()

    # ---- model ----
    start_ck = None
    if args.resume and os.path.exists(os.path.join(args.out, "ckpt.pt")):
        start_ck = torch.load(os.path.join(args.out, "ckpt.pt"), map_location="cpu", weights_only=False)
        cfg = ModelConfig(**start_ck["config"])
    elif args.init_from:
        init = torch.load(args.init_from, map_location="cpu", weights_only=False)
        cfg = ModelConfig(**init["config"])
    else:
        cfg = PRESETS[args.preset]
    assert cfg.vocab_size >= tok.get_vocab_size()
    model = GPT(cfg)
    if start_ck:
        model.load_state_dict(start_ck["model"])
    elif args.init_from:
        model.load_state_dict({k: v.float() for k, v in init["model"].items()})
    model.to(device)
    print(f"model: {model.num_params() / 1e6:.1f}M params ({model.num_params(True) / 1e6:.1f}M non-embedding), "
          f"{cfg.n_layer} layers, d={cfg.d_model}, ctx={cfg.seq_len}, device={device}", flush=True)

    lr_scale = args.lr_scale if args.stage == "pretrain" else args.lr_scale * 0.3
    opts = build_optimizers(model, args.optimizer, lr_scale=lr_scale)
    step, tokens, train_time = 0, 0, 0.0
    if start_ck:
        for o, s in zip(opts, start_ck["optim"]):
            o.load_state_dict(s)
        step, tokens, train_time = start_ck["step"], start_ck["tokens"], start_ck["train_time"]
        print(f"resumed at step {step}, {tokens / 1e6:.0f}M tokens, {train_time / 60:.1f} min", flush=True)

    use_compile = args.compile == "on" or (args.compile == "auto" and device == "cuda" and args.stage == "pretrain")
    fwd = model
    if use_compile:
        try:
            fwd = torch.compile(model)
        except Exception as e:
            print(f"torch.compile unavailable ({e!r}); running eagerly", flush=True)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else contextlib.nullcontext()

    # ---- data ----
    micro_bs = args.micro_bs or pick_micro_bs(device, cfg)
    if args.stage == "pretrain":
        from scoder.data import Mixture, build_val_sets

        mix_main = MIXES[args.mix]
        mix_cool = MIXES["anneal"] if args.mix == "pretrain" else mix_main
        sources = sorted(set(mix_main) | set(mix_cool))
        batch_tokens = args.batch_tokens or (262_144 if device == "cuda" else 8 * cfg.seq_len)
        accum = max(1, batch_tokens // (micro_bs * cfg.seq_len))
        print(f"data: {len(sources)} sources; {micro_bs} seqs x {accum} accum x {cfg.seq_len} = "
              f"{micro_bs * accum * cfg.seq_len:,} tokens/step", flush=True)
        val_sets = build_val_sets(tok, cfg.seq_len, sources)
        mixture = Mixture(tok, cfg.seq_len, sources, seed=args.seed + step)
    else:
        from scoder.data import SFTData

        sft = SFTData(args.sft_data, tok, cfg.seq_len, seed=args.seed)
        accum = 1
        val_sets = {}
        if not args.max_steps:  # a few passes over the small agent set, not as many as the clock allows
            args.max_steps = math.ceil(args.sft_epochs * len(sft.items) / micro_bs)
            print(f"SFT: {args.sft_epochs} epochs = {args.max_steps} steps (or {args.time_min} min, whichever first)")
    cooldown = args.cooldown_frac if args.cooldown_frac is not None else (COOLDOWN_FRAC if args.stage == "pretrain" else 1.0)
    warmup = args.warmup_steps if args.stage == "pretrain" else 10

    hub = HubPusher(args.hf_repo)
    budget = args.time_min * 60
    metrics_path = os.path.join(args.out, "metrics.jsonl")

    def log(rec):
        rec["time_min"] = round(train_time / 60, 2)
        print(json.dumps(rec), flush=True)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    @torch.no_grad()
    def evaluate():
        model.eval()
        res = {}
        for name, seqs in val_sets.items():
            losses = []
            for i in range(0, len(seqs), micro_bs):
                b = seqs[i : i + micro_bs].to(device)
                with amp:
                    losses.append(float(fwd(b[:, :-1], b[:, 1:])))
            res[name] = round(sum(losses) / len(losses), 4)
        model.train()
        if res:
            res["mean"] = round(sum(res.values()) / len(res), 4)
        return res

    def save(final=False):
        t = time.time()
        ck = {"model": model.state_dict(), "config": cfg.to_dict(), "step": step, "tokens": tokens,
              "train_time": train_time, "optim": [o.state_dict() for o in opts], "args": vars(args),
              "stage": args.stage}
        tmp = os.path.join(args.out, "ckpt.pt.tmp")
        torch.save(ck, tmp)
        os.replace(tmp, os.path.join(args.out, "ckpt.pt"))
        # small export for the Hub: bf16 weights + config + tokenizer + metrics
        hubdir = os.path.join(args.out, "hub")
        os.makedirs(hubdir, exist_ok=True)
        torch.save({"model": {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()},
                    "config": cfg.to_dict(), "step": step, "tokens": tokens, "stage": args.stage},
                   os.path.join(hubdir, f"{args.stage}.pt"))
        shutil.copy(TOKENIZER_PATH, os.path.join(hubdir, "tokenizer.json"))
        if os.path.exists(metrics_path):
            shutil.copy(metrics_path, os.path.join(hubdir, f"{args.stage}_metrics.jsonl"))
        hub.push(hubdir, f"{args.stage} step {step} ({tokens / 1e6:.0f}M tokens)" + (" final" if final else ""))
        return time.time() - t

    # ---- loop ----
    model.train()
    last_eval = last_ckpt = train_time
    t_log, tok_log = time.time(), tokens
    loop_start = time.time() - train_time
    while True:
        train_time = time.time() - loop_start
        # progress drives the LR schedule: whichever runs out first, time or steps
        progress = max(train_time / budget, step / args.max_steps if args.max_steps else 0.0)
        if progress >= 1:
            break
        mult = lr_mult(step, progress, warmup, cooldown)
        set_lr(opts, mult, muon_momentum=0.85 + 0.10 * min(1.0, step / 300))
        in_cooldown = args.stage == "pretrain" and progress >= 1 - cooldown
        loss_acc, step_tokens = 0.0, 0
        try:
            for _ in range(accum):
                if args.stage == "pretrain":
                    batch = mixture.get_batch(micro_bs, mix_cool if in_cooldown else mix_main).to(device, non_blocking=True)
                    x, y = batch[:, :-1], batch[:, 1:]
                else:
                    x, y = sft.get_batch(micro_bs)
                    x, y = x.to(device), y.to(device)
                with amp:
                    loss = fwd(x, y) / accum
                loss.backward()
                loss_acc += float(loss.detach())
                step_tokens += x.numel()
        except torch.cuda.OutOfMemoryError:
            # unattended run: shrink the micro-batch instead of crashing (only early on)
            if step > 5 or micro_bs == 1:
                raise
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            micro_bs //= 2
            accum *= 2
            print(f"out of GPU memory: micro-batch -> {micro_bs}, accumulation -> {accum}", flush=True)
            continue
        tokens += step_tokens
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in opts:
            o.step()
        model.zero_grad(set_to_none=True)
        step += 1

        if not math.isfinite(loss_acc):
            raise RuntimeError(f"loss became {loss_acc} at step {step}")
        if step % args.log_every == 0 or step == 1:
            now = time.time()
            rec = {"step": step, "loss": round(loss_acc, 4), "lr_mult": round(mult, 3),
                   "tok_per_s": int((tokens - tok_log) / max(1e-6, now - t_log)), "tokens_M": round(tokens / 1e6, 1),
                   "progress": round(progress, 3)}
            if args.stage == "pretrain":
                rec["phase"] = "anneal" if in_cooldown else "main"
                if step % (args.log_every * 10) == 0:
                    rec["data"] = mixture.stats()
            log(rec)
            t_log, tok_log = now, tokens
        # evaluation and checkpoints count against the budget: the budget is wall-clock money
        if val_sets and train_time - last_eval >= args.eval_every_min * 60:
            log({"step": step, "val": evaluate()})
            last_eval = train_time
        if train_time - last_ckpt >= args.ckpt_every_min * 60:
            save()
            last_ckpt = train_time

    if val_sets:
        log({"step": step, "val": evaluate(), "final": True})
    save(final=True)
    log({"done": True, "step": step, "tokens_M": round(tokens / 1e6, 1)})
    hub.wait()


if __name__ == "__main__":
    main()
