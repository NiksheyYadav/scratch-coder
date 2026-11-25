"""Streaming multi-source data: no preprocessing step and no big download before training.

Each source (a Stack-Edu language, or the agent sessions) gets a background thread
that reads parquet row groups straight from the Hugging Face Hub, tokenizes them and
packs the tokens into fixed-length sequences. The trainer then draws each sequence
from a source picked by the current mix weights. A source that falls behind is
skipped for that draw instead of stalling the GPU.
"""
from __future__ import annotations

import json
import os
import queue
import random
import threading
import time
from collections import Counter

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from .chat import EOT, PAD, encode_with_mask, render
from .config import SOURCES, STACK_EDU_REPO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "data", "stack_edu_manifest.json")
MAX_DOC_CHARS = 200_000  # truncate pathological giant files
BLOCK = 16 * 2**20

_fs = None


def hf_fs():
    global _fs
    if _fs is None:
        from huggingface_hub import HfFileSystem

        _fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))
    return _fs


def source_files(name: str) -> list[str]:
    """Remote parquet paths for a source, in a stable order."""
    kind, sel = SOURCES[name]
    if kind == "stack_edu":
        man = json.load(open(MANIFEST))
        return [f"datasets/{STACK_EDU_REPO}/{f['file']}" for f in man["files"] if sel in f.get("languages", [])]
    if kind == "agent":
        return sorted(hf_fs().glob(f"datasets/{sel}/**/*.parquet"))
    raise ValueError(kind)


def _open(path: str) -> pq.ParquetFile:
    for attempt in range(6):
        try:
            return pq.ParquetFile(hf_fs().open(path, "rb", block_size=BLOCK))
        except Exception as e:  # network hiccup: back off and retry
            err = e
            time.sleep(2 ** attempt)
    raise err


def _read_row_group(pf: pq.ParquetFile, rg: int, columns: list[str]):
    for attempt in range(6):
        try:
            return pf.read_row_group(rg, columns=columns)
        except Exception as e:
            err = e
            time.sleep(2 ** attempt)
    raise err


def _docs_from_table(name: str, table) -> list[str]:
    kind, sel = SOURCES[name]
    if kind == "stack_edu":
        table = table.filter(pc.equal(table.column("language"), sel))
        return [t[:MAX_DOC_CHARS] for t in table.column("text").to_pylist() if t]
    msgs = table.column("messages").to_pylist()
    return [render(m)[: MAX_DOC_CHARS * 4] for m in msgs if m]


def _columns(name: str) -> list[str]:
    return ["language", "text"] if SOURCES[name][0] == "stack_edu" else ["messages"]


def val_unit(name: str) -> tuple[str, int | None]:
    """The held-out (file, row_group) of a source; row_group None means the whole file.

    Stack-Edu files are sorted by language, so a language's first file starts with the
    previous language and ends with this one: its *last* row group is safely in-language.
    Agent files are single-source, so the last file is held out whole.
    """
    files = source_files(name)
    if SOURCES[name][0] == "agent":
        return files[-1], None
    return files[0], _open(files[0]).metadata.num_row_groups - 1


def _is_val(f: str, rg: int, vu: tuple[str, int | None]) -> bool:
    return f == vu[0] and (vu[1] is None or rg == vu[1])


def iter_source_documents_sample(name: str):
    """Documents from one training row group of a source (used to train the tokenizer)."""
    files = source_files(name)
    f = files[1] if len(files) > 2 else files[0]
    docs = _docs_from_table(name, _read_row_group(_open(f), 0, _columns(name)))
    random.Random(0).shuffle(docs)
    yield from docs


def _training_units(name: str, rng: random.Random):
    """Endless iterator over (file, row_group) training units, reshuffled each epoch."""
    files = source_files(name)
    if not files:
        raise RuntimeError(f"no files for source {name}")
    vu = val_unit(name)
    while True:
        order = files[:]
        rng.shuffle(order)
        for f in order:
            try:
                pf = _open(f)
            except Exception as e:  # skip this file for now; it comes round again next epoch
                print(f"[data] WARNING {name}: cannot open {f}: {e!r}", flush=True)
                time.sleep(30)
                continue
            rgs = [rg for rg in range(pf.metadata.num_row_groups) if not _is_val(f, rg, vu)]
            rng.shuffle(rgs)
            for rg in rgs:
                yield pf, f, rg


def val_docs(name: str, n_docs: int = 64) -> list[str]:
    f, rg = val_unit(name)
    docs = _docs_from_table(name, _read_row_group(_open(f), rg or 0, _columns(name)))
    return docs[:n_docs]


class SourceStream(threading.Thread):
    """Background producer: row groups -> documents -> tokens -> packed (seq_len+1) sequences."""

    def __init__(self, name: str, tok, seq_len: int, seed: int, maxsize: int = 512):
        super().__init__(daemon=True, name=f"stream-{name}")
        self.name_ = name
        self.tok = tok
        self.seq_len = seq_len
        self.rng = random.Random(seed)
        self.q: queue.Queue = queue.Queue(maxsize=maxsize)
        self.eot = tok.token_to_id(EOT)
        self.error: Exception | None = None
        self.docs_seen = 0
        self.tokens_made = 0

    def _tables(self):
        """Prefetch the next row group while the current one is being tokenized."""
        cols = _columns(self.name_)
        units = _training_units(self.name_, random.Random(self.rng.random()))
        pending: queue.Queue = queue.Queue(maxsize=1)

        def reader():
            fails = 0
            try:
                for pf, f, rg in units:
                    try:
                        pending.put(_read_row_group(pf, rg, cols))
                        fails = 0
                    except Exception as e:  # skip the unit; give up only after a long outage
                        fails += 1
                        print(f"[data] WARNING {self.name_}: row group {rg} of {f} failed ({fails}): {e!r}", flush=True)
                        if fails >= 20:
                            raise
                        time.sleep(30)
            except Exception as e:  # surface in the consumer
                pending.put(e)

        threading.Thread(target=reader, daemon=True).start()
        while True:
            item = pending.get()
            if isinstance(item, Exception):
                raise item
            yield item

    def run(self):
        try:
            rest = np.zeros(0, dtype=np.int32)
            S, L = self.seq_len, self.seq_len + 1
            eot = np.array([self.eot], dtype=np.int32)
            for table in self._tables():
                docs = _docs_from_table(self.name_, table)
                self.rng.shuffle(docs)
                for i in range(0, len(docs), 256):
                    chunk = docs[i : i + 256]
                    parts = [rest]
                    for enc in self.tok.encode_batch(chunk, add_special_tokens=False):
                        parts.append(np.asarray(enc.ids, dtype=np.int32))
                        parts.append(eot)
                    arr = np.concatenate(parts)
                    self.docs_seen += len(chunk)
                    n = (len(arr) - 1) // S
                    for j in range(n):  # consecutive windows share one boundary token
                        self.q.put(arr[j * S : j * S + L].copy())
                    self.tokens_made += n * S
                    rest = arr[n * S :]
        except Exception as e:
            self.error = e
            raise


class Mixture:
    """Draws training sequences from several SourceStreams according to mix weights."""

    def __init__(self, tok, seq_len: int, sources: list[str], seed: int = 1234):
        self.streams = {s: SourceStream(s, tok, seq_len, seed + i) for i, s in enumerate(sources)}
        for st in self.streams.values():
            st.start()
        self.rng = np.random.default_rng(seed)
        self.drawn = Counter()
        self.substituted = 0

    def get_batch(self, batch_size: int, weights: dict[str, float]) -> torch.Tensor:
        names = [n for n in weights if weights[n] > 0 and n in self.streams]
        p = np.array([weights[n] for n in names], dtype=np.float64)
        p /= p.sum()
        picks = self.rng.choice(names, size=batch_size, p=p)
        out = []
        for name in picks:
            seq = self._take(str(name), names)
            out.append(seq)
        return torch.from_numpy(np.stack(out)).long()

    def _take(self, name: str, names: list[str]):
        st = self.streams[name]
        if st.error is not None:
            raise RuntimeError(f"data source {name} failed: {st.error!r}")
        try:
            seq = st.q.get(timeout=0.5)
            self.drawn[name] += 1
            return seq
        except queue.Empty:
            pass
        # source is behind: take from whichever source has data rather than idling the GPU
        deadline = time.time() + 600
        while time.time() < deadline:
            for n in map(str, self.rng.permutation(names)):
                try:
                    seq = self.streams[n].q.get_nowait()
                    self.drawn[n] += 1
                    self.substituted += 1
                    return seq
                except queue.Empty:
                    continue
            errs = {n: s.error for n, s in self.streams.items() if s.error is not None}
            if errs:
                raise RuntimeError(f"data sources failed: {errs}")
            time.sleep(0.2)
        raise TimeoutError("no data from any source for 10 minutes")

    def stats(self) -> dict:
        total = sum(self.drawn.values()) or 1
        return {
            "share": {k: round(v / total, 3) for k, v in sorted(self.drawn.items())},
            "substituted": self.substituted,
            "queued": {k: s.q.qsize() for k, s in self.streams.items()},
        }


def build_val_sets(tok, seq_len: int, sources: list[str], n_seqs: int = 8) -> dict[str, torch.Tensor]:
    """Fixed validation sequences per source, from held-out row groups."""
    from concurrent.futures import ThreadPoolExecutor

    eot = tok.token_to_id(EOT)
    out = {}
    with ThreadPoolExecutor(8) as ex:  # fetch all held-out row groups in parallel
        docs = dict(zip(sources, ex.map(lambda s: val_docs(s, 256), sources)))
    for name in sources:
        buf: list[int] = []
        for enc in tok.encode_batch(docs[name], add_special_tokens=False):
            buf.extend(enc.ids)
            buf.append(eot)
            if len(buf) >= n_seqs * (seq_len + 1):
                break
        n = min(n_seqs, len(buf) // (seq_len + 1))
        if n == 0:
            continue
        arr = np.asarray(buf[: n * (seq_len + 1)], dtype=np.int64).reshape(n, seq_len + 1)
        out[name] = torch.from_numpy(arr)
    return out


class SFTData:
    """Agent fine-tuning examples (JSONL of {"messages": [...]}) with assistant-only loss masks."""

    def __init__(self, path: str, tok, seq_len: int, seed: int = 0):
        self.pad = tok.token_to_id(PAD)
        self.seq_len = seq_len
        self.items = []
        too_long = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                ids, mask = encode_with_mask(tok, ex["messages"])
                if len(ids) > seq_len + 1:
                    # keep the end of the conversation, where the assistant turns are
                    too_long += 1
                    ids, mask = ids[-(seq_len + 1):], mask[-(seq_len + 1):]
                if sum(mask) == 0:
                    continue
                self.items.append((ids, mask))
        if not self.items:
            raise RuntimeError(f"no usable SFT examples in {path}")
        self.rng = random.Random(seed)
        self.order: list[int] = []
        lens = sorted(len(i) for i, _ in self.items)
        print(f"SFT data: {len(self.items)} examples, median {lens[len(lens) // 2]} tokens, "
              f"{too_long} left-truncated to {seq_len + 1}")

    def get_batch(self, batch_size: int):
        if len(self.order) < batch_size:
            self.order = list(range(len(self.items)))
            self.rng.shuffle(self.order)
        idx = [self.order.pop() for _ in range(batch_size)]
        L = max(len(self.items[i][0]) for i in idx)
        x = torch.full((batch_size, L), self.pad, dtype=torch.long)
        m = torch.zeros((batch_size, L), dtype=torch.long)
        for r, i in enumerate(idx):
            ids, mask = self.items[i]
            x[r, : len(ids)] = torch.tensor(ids)
            m[r, : len(mask)] = torch.tensor(mask)
        # targets: next token, only where the *target* token is assistant content
        y = x[:, 1:].clone()
        y[m[:, 1:] == 0] = -1
        return x[:, :-1], y
