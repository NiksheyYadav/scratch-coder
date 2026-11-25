"""Print a completion from a checkpoint.

    python sample.py --ckpt out/pretrain/ckpt.pt --prompt "def fibonacci(n):"
"""
import argparse

import torch

from scoder.chat import END, EOT
from scoder.model import load_checkpoint
from scoder.tokenizer import load_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="def fibonacci(n):")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_checkpoint(args.ckpt, device)
    tok = load_tokenizer()
    g = torch.Generator(device=device).manual_seed(args.seed)
    stops = {tok.token_to_id(EOT), tok.token_to_id(END)}
    ids = tok.encode(args.prompt).ids
    for i in range(args.n):
        out = model.generate(ids, args.max_new_tokens, stop_ids=stops, temperature=args.temperature, generator=g)
        print(f"--- sample {i + 1} (step {ck.get('step')}, {ck.get('tokens', 0) / 1e6:.0f}M tokens) ---")
        print(args.prompt + tok.decode(out, skip_special_tokens=False))


if __name__ == "__main__":
    main()
