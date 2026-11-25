"""Byte-level BPE tokenizer trained on our own multi-language data mix.

Train (once, on a laptop is fine):  python -m scoder.tokenizer --out tokenizer/tokenizer.json
"""
from __future__ import annotations

import argparse
import os

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

from .chat import SPECIAL_TOKENS

# GPT-4 style split: keeps runs of whitespace (indentation) together and caps digit runs at 3.
SPLIT_PATTERN = (
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tokenizer", "tokenizer.json")


def load_tokenizer(path: str = DEFAULT_PATH) -> Tokenizer:
    return Tokenizer.from_file(path)


def build_empty() -> Tokenizer:
    tok = Tokenizer(models.BPE(byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(SPLIT_PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def sample_texts(bytes_per_source: int, sources: list[str]):
    """Yield training text: one row group per Stack-Edu language plus rendered agent sessions."""
    from .data import iter_source_documents_sample

    for name in sources:
        total = 0
        for text in iter_source_documents_sample(name):
            yield text
            total += len(text)
            if total >= bytes_per_source:
                break
        print(f"  tokenizer sample: {name:<11} {total / 1e6:6.1f} MB", flush=True)


def main():
    from .config import SOURCES

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_PATH)
    ap.add_argument("--vocab_size", type=int, default=16384)
    ap.add_argument("--mb_per_source", type=float, default=12.0)
    args = ap.parse_args()

    tok = build_empty()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    texts = sample_texts(int(args.mb_per_source * 1e6), list(SOURCES))
    tok.train_from_iterator(texts, trainer=trainer)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tok.save(args.out)
    assert tok.token_to_id(SPECIAL_TOKENS[0]) == 0
    print(f"saved {args.out} (vocab {tok.get_vocab_size()})")


if __name__ == "__main__":
    main()
