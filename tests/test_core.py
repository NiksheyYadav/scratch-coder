"""Fast offline tests for the core pieces. No network, no GPU, no real keys."""
import pytest
import torch

from scoder import chat
from scoder.config import MIXES, PRESETS, SOURCES
from scoder.model import GPT, load_checkpoint
from scoder.optim import build_optimizers, set_lr
from scoder.tokenizer import load_tokenizer


# ------------------------------------------------------------------ config
def test_every_mix_is_a_valid_distribution_over_known_sources():
    for name, mix in MIXES.items():
        assert abs(sum(mix.values()) - 1.0) < 1e-6, name
        assert set(mix) <= set(SOURCES), name


def test_presets_have_consistent_head_dims():
    for name, cfg in PRESETS.items():
        assert cfg.d_model % cfg.n_head == 0, name
        assert (cfg.d_model // cfg.n_head) % 2 == 0, f"{name}: rotary needs an even head_dim"


# --------------------------------------------------------------- tokenizer
@pytest.fixture(scope="module")
def tok():
    return load_tokenizer()


def test_tokenizer_knows_all_special_tokens(tok):
    for t in chat.SPECIAL_TOKENS:
        assert tok.token_to_id(t) is not None, t


def test_tokenizer_roundtrips_code(tok):
    text = "def add(a, b):\n    return a + b\n\n# éè ✓\n"
    assert tok.decode(tok.encode(text).ids, skip_special_tokens=False) == text


# -------------------------------------------------------------------- chat
def test_parse_actions_takes_each_bash_block():
    one = "THOUGHT: ok\n\n```bash\nls -la\n```"
    two = one + "\n\n```bash\npwd\n```"
    assert chat.parse_actions(one) == ["ls -la"]
    assert chat.parse_actions(two) == ["ls -la", "pwd"]
    assert chat.parse_actions("no code here") == []


def test_format_observation_elides_long_output():
    out = chat.format_observation(1, "x" * 10000, limit=100)
    assert "<returncode>1</returncode>" in out and "characters elided" in out
    assert len(out) < 400


def test_mask_trains_only_on_assistant_text(tok):
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "THOUGHT: done"},
    ]
    ids, mask = chat.encode_with_mask(tok, msgs)
    assert len(ids) == len(mask) and sum(mask) > 0
    trained = tok.decode([i for i, m in zip(ids, mask) if m], skip_special_tokens=False)
    assert "THOUGHT: done" in trained and "SYS" not in trained and "do it" not in trained
    assert chat.END in trained  # the END token is trained so the model learns to stop


def test_render_ends_with_generation_header_when_asked():
    text = chat.render([{"role": "user", "content": "hi"}], add_generation_prompt=True)
    assert text.endswith(chat.header("assistant"))
    assert not text.endswith(chat.EOT)


# ------------------------------------------------------------------- model
@pytest.fixture(scope="module")
def nano():
    torch.manual_seed(0)
    m = GPT(PRESETS["nano"])
    torch.nn.init.normal_(m.lm_head.weight, std=0.05)  # zero-init head makes every logit equal
    for b in m.blocks:
        torch.nn.init.normal_(b.attn.proj.weight, std=0.05)
        torch.nn.init.normal_(b.mlp.proj.weight, std=0.05)
    return m.eval()


def test_untrained_loss_is_near_log_vocab():
    torch.manual_seed(0)
    m = GPT(PRESETS["nano"])  # default init: zero head => uniform prediction
    x = torch.randint(0, m.cfg.vocab_size, (2, 16))
    loss = m(x, targets=x)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert abs(float(loss.detach()) - torch.log(torch.tensor(float(m.cfg.vocab_size))).item()) < 0.05


def test_ignore_index_is_respected():
    torch.manual_seed(0)
    m = GPT(PRESETS["nano"])
    x = torch.randint(0, m.cfg.vocab_size, (1, 8))
    t = x.clone()
    t[:, :4] = -1
    assert torch.isfinite(m(x, targets=t))
    assert float(m(x, targets=torch.full_like(x, -1))) == 0.0  # all masked: no NaN


def test_kv_cache_generation_matches_uncached_greedy(nano):
    ids = [5, 17, 42, 7, 99]
    cached = nano.generate(ids, 12, temperature=0)
    seq = list(ids)
    with torch.no_grad():
        for _ in range(12):
            seq.append(int(nano(torch.tensor([seq]))[0, -1].argmax()))
    assert cached == seq[len(ids):]


def test_generate_stops_at_stop_id_and_checks_context(nano):
    first = nano.generate([1, 2, 3], 1, temperature=0)[0]
    out = nano.generate([1, 2, 3], 50, stop_ids={first}, temperature=0)
    assert out == [first]
    with pytest.raises(ValueError):
        nano.generate(list(range(nano.cfg.seq_len)), 4)


def test_checkpoint_roundtrip(tmp_path, nano):
    path = tmp_path / "ck.pt"
    torch.save({"config": nano.cfg.to_dict(), "model": nano.state_dict()}, path)
    loaded, _ = load_checkpoint(str(path))
    x = torch.tensor([[1, 2, 3, 4]])
    assert torch.allclose(nano(x), loaded(x), atol=1e-6)


# --------------------------------------------------------------- optimizer
@pytest.mark.parametrize("kind", ["muon", "adamw"])
def test_optimizer_step_reduces_loss_on_a_fixed_batch(kind):
    torch.manual_seed(0)
    m = GPT(PRESETS["nano"])
    x = torch.randint(0, m.cfg.vocab_size, (4, 32))
    opts = build_optimizers(m, kind)
    before = float(m(x, targets=x))
    for _ in range(5):
        for o in opts:
            o.zero_grad(set_to_none=True)
        m(x, targets=x).backward()
        for o in opts:
            o.step()
    assert float(m(x, targets=x)) < before


def test_set_lr_scales_from_base_lr():
    m = GPT(PRESETS["nano"])
    opts = build_optimizers(m, "muon")
    set_lr(opts, 0.5)
    for o in opts:
        for g in o.param_groups:
            assert g["lr"] == pytest.approx(g["base_lr"] * 0.5)
