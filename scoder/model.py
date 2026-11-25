"""A small modern decoder-only transformer.

Choices follow the 2024-2025 "speedrun" line of work rather than GPT-2:
RMSNorm without learned scales, rotary position embeddings, QK-norm,
squared-ReLU MLP, zero-initialised output projections, an untied output head
and tanh soft-capped logits. Generation uses a KV cache.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def rms(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class Rotary(nn.Module):
    def __init__(self, head_dim: int, max_len: int, base: float = 10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_len, dtype=torch.float32)
        f = torch.outer(t, inv)
        self.register_buffer("cos", f.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin", f.sin()[None, :, None, :], persistent=False)

    def forward(self, x: torch.Tensor, pos: int) -> torch.Tensor:  # x: (B, T, H, D)
        T = x.size(1)
        cos = self.cos[:, pos : pos + T].to(x.dtype)
        sin = self.sin[:, pos : pos + T].to(x.dtype)
        d = x.size(-1) // 2
        x1, x2 = x[..., :d], x[..., d:]
        return torch.cat([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, rotary: Rotary):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rotary = rotary

    def forward(self, x, cache: dict | None = None, pos: int = 0):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).unbind(2)
        q, k = rms(q), rms(k)  # QK-norm
        q, k = self.rotary(q, pos), self.rotary(k, pos)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, H, T, D)
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        causal = k.size(2) == T  # full prefill is causal; one-token decode attends to all past
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.mlp_mult * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.mlp_mult * cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        return self.proj(F.relu(self.fc(x)).square())


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, rotary: Rotary):
        super().__init__()
        self.attn = Attention(cfg, rotary)
        self.mlp = MLP(cfg)

    def forward(self, x, cache=None, pos=0):
        x = x + self.attn(rms(x), cache, pos)
        return x + self.mlp(rms(x))


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        rotary = Rotary(cfg.d_model // cfg.n_head, cfg.seq_len)
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, rotary) for _ in range(cfg.n_layer)])
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self._init()

    def _init(self):
        nn.init.normal_(self.wte.weight, std=1.0)
        for b in self.blocks:
            for lin in (b.attn.qkv, b.mlp.fc):
                nn.init.normal_(lin.weight, std=0.5 / math.sqrt(lin.in_features))
            nn.init.zeros_(b.attn.proj.weight)
            nn.init.zeros_(b.mlp.proj.weight)
        nn.init.zeros_(self.lm_head.weight)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - self.wte.weight.numel() if non_embedding else n

    def forward(self, idx, targets=None, caches=None, pos: int = 0):
        x = rms(self.wte(idx))
        for i, b in enumerate(self.blocks):
            x = b(x, None if caches is None else caches[i], pos)
        x = rms(x)
        if targets is None:
            x = x[:, -1:]  # inference: only the last position is needed
        logits = self.lm_head(x).float()
        c = self.cfg.logit_softcap
        logits = c * torch.tanh(logits / c)
        if targets is None:
            return logits
        t = targets.reshape(-1)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), t, ignore_index=-1, reduction="sum")
        return loss / (t != -1).sum().clamp(min=1)

    @torch.no_grad()
    def generate(self, ids: list[int], max_new_tokens: int, stop_ids: set[int] | None = None,
                 temperature: float = 0.7, top_k: int = 40, top_p: float = 0.95,
                 repetition_penalty: float = 1.0, generator: torch.Generator | None = None) -> list[int]:
        """Sample a continuation with a KV cache. Stops at any token in stop_ids."""
        dev = next(self.parameters()).device
        max_ctx = self.cfg.seq_len
        budget = max_ctx - len(ids)
        if budget <= 0:
            raise ValueError(f"prompt of {len(ids)} tokens does not fit the {max_ctx}-token context")
        max_new_tokens = min(max_new_tokens, budget)
        caches = [dict() for _ in self.blocks]
        x = torch.tensor([ids], device=dev)
        logits = self(x, caches=caches, pos=0)[0, -1]
        out: list[int] = []
        pos = len(ids)
        for _ in range(max_new_tokens):
            if repetition_penalty != 1.0 and out:
                seen = torch.tensor(sorted(set(out[-256:])), device=dev)
                l = logits[seen]
                logits[seen] = torch.where(l > 0, l / repetition_penalty, l * repetition_penalty)
            if temperature <= 0:
                nxt = int(logits.argmax())
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                if top_k:
                    v, i = probs.topk(min(top_k, probs.numel()))
                    probs = torch.zeros_like(probs).scatter_(0, i, v)
                if top_p < 1.0:
                    sp, si = probs.sort(descending=True)
                    keep = sp.cumsum(0) - sp <= top_p * sp.sum()
                    probs = torch.zeros_like(probs).scatter_(0, si[keep], sp[keep])
                nxt = int(torch.multinomial(probs / probs.sum(), 1, generator=generator))
            out.append(nxt)
            if stop_ids and nxt in stop_ids:
                break
            logits = self(torch.tensor([[nxt]], device=dev), caches=caches, pos=pos)[0, -1]
            pos += 1
        return out


def load_checkpoint(path: str, device: str = "cpu") -> tuple[GPT, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ck["config"])
    model = GPT(cfg)
    model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    model.to(device).eval()
    return model, ck
