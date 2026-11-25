"""Optimizers: Muon for the transformer's weight matrices, Adam for embeddings and head.

Muon (Jordan et al., 2024) orthogonalises each matrix update with a few Newton-Schulz
iterations. It reached the same loss with markedly fewer tokens than AdamW in the
nanoGPT speedrun, which matters when the whole budget is $1.81.
"""
from __future__ import annotations

import torch


def _newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class MuonFallback(torch.optim.Optimizer):
    """Minimal Muon, used only when torch.optim.Muon is unavailable (PyTorch < 2.9)."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.0, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      weight_decay=weight_decay, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                buf = st.setdefault("momentum_buffer", torch.zeros_like(p.grad))
                buf.mul_(g["momentum"]).add_(p.grad)
                upd = p.grad.add(buf, alpha=g["momentum"]) if g["nesterov"] else buf
                upd = _newton_schulz(upd, g["ns_steps"]).to(p.dtype)
                if g["weight_decay"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(upd, alpha=-g["lr"] * max(1.0, p.size(0) / p.size(1)) ** 0.5)


def build_optimizers(model, kind: str, lr_scale: float = 1.0, muon_lr=0.02, embed_lr=0.3, head_lr=0.004,
                     adamw_lr=3e-3, weight_decay=0.0):
    cuda = next(model.parameters()).is_cuda
    matrices = [p for p in model.blocks.parameters() if p.ndim == 2]
    others = [p for p in model.blocks.parameters() if p.ndim != 2]
    assert not others, "all block parameters are expected to be matrices"
    opts = []
    if kind == "muon":
        adam = torch.optim.Adam(
            [dict(params=[model.wte.weight], lr=embed_lr * lr_scale),
             dict(params=[model.lm_head.weight], lr=head_lr * lr_scale)],
            betas=(0.8, 0.95), eps=1e-10, fused=cuda)
        Muon = getattr(torch.optim, "Muon", None) or MuonFallback
        muon = Muon(matrices, lr=muon_lr * lr_scale, momentum=0.95, nesterov=True, weight_decay=weight_decay)
        opts = [adam, muon]
    elif kind == "adamw":
        decay = matrices + [model.lm_head.weight]
        opts = [torch.optim.AdamW(
            [dict(params=decay, weight_decay=0.1), dict(params=[model.wte.weight], weight_decay=0.0)],
            lr=adamw_lr * lr_scale, betas=(0.9, 0.95), eps=1e-8, fused=cuda)]
    else:
        raise ValueError(kind)
    for o in opts:
        for g in o.param_groups:
            g["base_lr"] = g["lr"]
    return opts


def set_lr(opts, mult: float, muon_momentum: float | None = None):
    for o in opts:
        for g in o.param_groups:
            g["lr"] = g["base_lr"] * mult
            if muon_momentum is not None and "momentum" in g and "betas" not in g:
                g["momentum"] = muon_momentum
