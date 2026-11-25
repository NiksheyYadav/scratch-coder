"""Read a checkpoint's weights: statistics, per-head and per-neuron summaries, raw dumps,
and activation probes that compare two sets of model replies (e.g. runaway vs normal).

    # overview of every tensor, heads and MLP neurons
    python -m wsp.inspect_weights --ckpt out/sft/ckpt.pt --out out/wsp/report.json

    # raw numbers of one tensor slice, as text an LLM can read
    python -m wsp.inspect_weights --ckpt out/sft/ckpt.pt --dump blocks.0.mlp.fc.weight --rows 0:8

    # find the residual direction that separates failing replies from good ones
    python -m wsp.inspect_weights --ckpt out/sft/ckpt.pt --probe out/eval/episodes.jsonl --out out/wsp/report.json
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from scoder.chat import header
from scoder.model import load_checkpoint
from scoder.tokenizer import load_tokenizer


def tensor_stats(w: torch.Tensor) -> dict:
    w = w.float()
    s = {"shape": list(w.shape), "rms": round(float(w.pow(2).mean().sqrt()), 5),
         "max_abs": round(float(w.abs().max()), 5), "frac_zero": round(float((w == 0).float().mean()), 5)}
    if w.ndim == 2 and min(w.shape) > 1:
        sv = torch.linalg.svdvals(w)
        s["top_singular_values"] = [round(float(x), 4) for x in sv[:5]]
        s["stable_rank"] = round(float(sv.pow(2).sum() / sv[0].pow(2)), 2) if sv[0] > 0 else 0.0
    return s


def head_summary(model) -> list[dict]:
    cfg = model.cfg
    D = cfg.d_model // cfg.n_head
    out = []
    for li, b in enumerate(model.blocks):
        q, k, v = b.attn.qkv.weight.detach().float().view(3, cfg.n_head, D, cfg.d_model).unbind(0)
        o = b.attn.proj.weight.detach().float().view(cfg.d_model, cfg.n_head, D)
        for h in range(cfg.n_head):
            out.append({"layer": li, "head": h, "q_norm": round(float(q[h].norm()), 4),
                        "k_norm": round(float(k[h].norm()), 4), "v_norm": round(float(v[h].norm()), 4),
                        "o_norm": round(float(o[:, h].norm()), 4),
                        "ov_norm": round(float((o[:, h] @ v[h]).norm()), 4)})
    return out


def neuron_summary(model, top: int = 10) -> list[dict]:
    out = []
    for li, b in enumerate(model.blocks):
        win = b.mlp.fc.weight.detach().float().norm(dim=1)     # per neuron input norm
        wout = b.mlp.proj.weight.detach().float().norm(dim=0)  # per neuron output norm
        score = win * wout
        idx = score.argsort(descending=True)[:top]
        out.append({"layer": li, "n_neurons": int(win.numel()),
                    "near_dead": int((score < 1e-3 * score.max()).sum()),
                    "top_neurons": [{"neuron": int(i), "in_norm": round(float(win[i]), 4),
                                     "out_norm": round(float(wout[i]), 4)} for i in idx]})
    return out


@torch.no_grad()
def capture(model, tok, texts: list[str], max_tokens: int = 512):
    """Mean residual stream (after each block) and mean MLP neuron activation over the given texts."""
    L = len(model.blocks)
    resid = [None] * L
    neur = [None] * L
    count = 0
    hooks = []
    store = {}

    def mk_block_hook(i):
        def f(mod, inp, out):
            store[("r", i)] = out[0].float()
        return f

    def mk_fc_hook(i):
        def f(mod, inp, out):
            store[("n", i)] = torch.relu(out[0].float()).square()
        return f

    for i, b in enumerate(model.blocks):
        hooks.append(b.register_forward_hook(mk_block_hook(i)))
        hooks.append(b.mlp.fc.register_forward_hook(mk_fc_hook(i)))
    dev = next(model.parameters()).device
    try:
        for t in texts:
            ids = tok.encode(header("assistant") + t).ids[:max_tokens]
            if len(ids) < 2:
                continue
            model(torch.tensor([ids], device=dev))
            for i in range(L):
                r, n = store[("r", i)].sum(0), store[("n", i)].sum(0)
                resid[i] = r if resid[i] is None else resid[i] + r
                neur[i] = n if neur[i] is None else neur[i] + n
            count += len(ids)
    finally:
        for h in hooks:
            h.remove()
    if count == 0:
        raise ValueError("no usable texts to probe")
    return [r / count for r in resid], [n / count for n in neur], count


def replies_from_episodes(path: str) -> tuple[list[str], list[str]]:
    """Split assistant replies into (bad, good): bad = no single bash block or very long/looping."""
    from scoder.chat import parse_actions

    bad, good = [], []
    for line in open(path, encoding="utf-8"):
        ep = json.loads(line)
        for m in ep["messages"]:
            if m["role"] != "assistant":
                continue
            text = m["content"]
            words = text.split()
            repetitive = len(words) > 40 and len(set(words)) / len(words) < 0.3
            (bad if len(parse_actions(text)) != 1 or repetitive else good).append(text)
    return bad, good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default=None, help="parameter name to print as raw numbers")
    ap.add_argument("--rows", default="0:8")
    ap.add_argument("--cols", default="0:16")
    ap.add_argument("--probe", default=None, help="episodes.jsonl from agent.eval")
    ap.add_argument("--max_texts", type=int, default=200)
    args = ap.parse_args()

    model, ck = load_checkpoint(args.ckpt, "cpu")
    params = dict(model.named_parameters())
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.dump:
        w = params[args.dump].detach().float()
        r0, r1 = (int(x) for x in args.rows.split(":"))
        if w.ndim == 1:
            print(" ".join(f"{x:.3f}" for x in w[r0:r1].tolist()))
            return
        c0, c1 = (int(x) for x in args.cols.split(":"))
        for r in range(r0, min(r1, w.shape[0])):
            print(f"{args.dump}[{r}, {c0}:{c1}] " + " ".join(f"{x:+.3f}" for x in w[r, c0:c1].tolist()))
        return

    report = {"checkpoint": args.ckpt, "step": ck.get("step"), "config": ck["config"],
              "tensors": {n: tensor_stats(p.detach()) for n, p in params.items()},
              "heads": head_summary(model), "neurons": neuron_summary(model)}

    if args.probe:
        tok = load_tokenizer()
        bad, good = replies_from_episodes(args.probe)
        report["probe"] = {"bad_replies": len(bad), "good_replies": len(good)}
        if bad and good:
            rb, nb, _ = capture(model, tok, bad[: args.max_texts])
            rg, ng, _ = capture(model, tok, good[: args.max_texts])
            dirs = {}
            layers = []
            for i in range(len(model.blocks)):
                d = rb[i] - rg[i]
                dirs[f"layer{i}"] = d / (d.norm() + 1e-8)
                diff = nb[i] - ng[i]
                top = diff.abs().argsort(descending=True)[:10]
                layers.append({"layer": i, "resid_diff_norm": round(float(d.norm()), 4),
                               "neurons_most_different": [{"neuron": int(j), "bad_minus_good": round(float(diff[j]), 5)}
                                                          for j in top]})
            report["probe"]["layers"] = layers
            if args.out:
                dpath = os.path.splitext(args.out)[0] + "_directions.pt"
                torch.save(dirs, dpath)
                report["probe"]["directions_file"] = dpath
        else:
            report["probe"]["note"] = "need both bad and good replies to compare"

    text = json.dumps(report, indent=1)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}")
    else:
        print(text[:20000])


if __name__ == "__main__":
    main()
