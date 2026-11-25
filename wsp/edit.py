"""Apply direct weight edits to a checkpoint (no gradients, no training).

    python -m wsp.edit --ckpt out/sft/ckpt.pt --edits edits.json --out out/wsp/edited.pt

edits.json is a list of operations, applied in order:
  {"op": "scale", "param": "blocks.3.mlp.proj.weight", "cols": [12, 40], "factor": 0.0}
        multiply whole rows/cols (or the whole tensor) by a factor; cols of mlp.proj = MLP neurons
  {"op": "set", "param": "blocks.0.mlp.fc.weight", "index": [5, 7], "value": 0.25}
  {"op": "ablate_head", "layer": 2, "head": 1}
        silence one attention head by zeroing its slice of the output projection
  {"op": "project_out", "param": "blocks.5.mlp.proj.weight", "direction": "out/wsp/report_directions.pt:layer5",
   "strength": 1.0, "cols": [optional neuron list]}
        remove a residual-stream direction u from what this matrix writes: W <- W - s * u (u^T W)
        (the operation used to remove the "loop direction" in the MIT weight-space programming post)
  {"op": "add_direction", "param": "blocks.5.mlp.proj.weight", "direction": "file.pt:key", "cols": [3], "scale": 0.5}
        make chosen neurons also write along direction u
"""
from __future__ import annotations

import argparse
import json
import os

import torch


def _direction(spec: str, dim: int) -> torch.Tensor:
    path, key = spec.rsplit(":", 1)
    u = torch.load(path, map_location="cpu")[key].float().flatten()
    assert u.numel() == dim, f"direction has {u.numel()} dims, expected {dim}"
    return u / (u.norm() + 1e-8)


def apply_edits(state: dict, cfg: dict, edits: list[dict]) -> list[dict]:
    log = []
    for e in edits:
        op = e["op"]
        if op == "ablate_head":
            name = f"blocks.{e['layer']}.attn.proj.weight"
            D = cfg["d_model"] // cfg["n_head"]
            e = {"op": "scale", "param": name, "cols": list(range(e["head"] * D, (e["head"] + 1) * D)), "factor": 0.0,
                 "_from": e}
            op = "scale"
        W = state[e["param"]]
        before = W.clone()
        Wf = W.float()
        if op == "scale":
            if "rows" in e:
                Wf[e["rows"]] *= e["factor"]
            elif "cols" in e:
                Wf[:, e["cols"]] *= e["factor"]
            else:
                Wf *= e["factor"]
        elif op == "set":
            Wf[tuple(e["index"])] = e["value"]
        elif op == "project_out":
            u = _direction(e["direction"], Wf.shape[0])
            cols = e.get("cols")
            sub = Wf if cols is None else Wf[:, cols]
            sub = sub - e.get("strength", 1.0) * torch.outer(u, u @ sub)
            if cols is None:
                Wf = sub
            else:
                Wf[:, cols] = sub
        elif op == "add_direction":
            u = _direction(e["direction"], Wf.shape[0])
            for c in e["cols"]:
                Wf[:, c] += e.get("scale", 1.0) * u * Wf[:, c].norm()
        else:
            raise ValueError(f"unknown op {op}")
        state[e["param"]] = Wf.to(W.dtype)
        changed = int((state[e["param"]] != before).sum())
        log.append({"edit": e, "weights_changed": changed, "delta_norm": round(float((state[e["param"]].float() - before.float()).norm()), 5)})
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--edits", required=True, help="JSON file with a list of edit operations")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    edits = json.load(open(args.edits))
    state = {k: v.clone() for k, v in ck["model"].items()}
    log = apply_edits(state, ck["config"], edits)
    total = sum(v.numel() for v in state.values())
    changed = sum(l["weights_changed"] for l in log)
    for l in log:
        print(json.dumps(l))
    print(f"changed {changed:,} of {total:,} weights ({100 * changed / total:.4f}%)")
    ck["model"] = state
    ck.pop("optim", None)  # optimizer state no longer matches the edited weights
    ck.setdefault("wsp_edits", []).append({"from": args.ckpt, "edits": edits, "log": log})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(ck, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
