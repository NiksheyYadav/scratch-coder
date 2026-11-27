"""Hugging Face token handling. All keys here are dummies; nothing touches the network."""
import importlib.util
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
GOOD = "hf_" + "A" * 34  # dummy, shaped like a real token


def load_preflight():
    spec = importlib.util.spec_from_file_location("hub_preflight", ROOT / "scripts" / "hub_preflight.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class HubError(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.response = types.SimpleNamespace(status_code=status)


@pytest.fixture
def fake_hub(monkeypatch):
    """Replace huggingface_hub.HfApi with a recorder. `behaviour` picks what whoami raises."""
    state = {"tokens": [], "whoami_error": None, "write_error": None}

    class FakeApi:
        def __init__(self, token=None):
            state["tokens"].append(token)

        def whoami(self):
            if state["whoami_error"]:
                raise state["whoami_error"]
            return {"name": "tester"}

        def create_repo(self, *a, **k):
            if state["write_error"]:
                raise state["write_error"]

        def upload_file(self, **k):
            if state["write_error"]:
                raise state["write_error"]

    mod = types.ModuleType("huggingface_hub")
    mod.HfApi = FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    return state


# ------------------------------------------------------------ clean_token
@pytest.mark.parametrize("raw", [
    GOOD, GOOD + "\n", GOOD + "\r\n", "  " + GOOD + "  ", "﻿" + GOOD, f'"{GOOD}"', f"'{GOOD}'",
    "Bearer " + GOOD, "bearer " + GOOD, GOOD[:10] + "\n" + GOOD[10:],
])
def test_clean_token_recovers_the_real_token_from_copy_paste_damage(raw):
    from scoder.hub import clean_token

    assert clean_token(raw) == GOOD


@pytest.mark.parametrize("raw", [None, "", "   ", "\n", '""'])
def test_clean_token_returns_none_for_empty_input(raw):
    from scoder.hub import clean_token

    assert clean_token(raw) is None


def test_token_problem_flags_other_services_keys_without_echoing_them():
    from scoder.hub import token_problem

    assert token_problem(GOOD) is None
    for wrong in ("cnx_" + "x" * 32, "nvapi-" + "x" * 40, "sk-" + "x" * 40, "ghp_" + "x" * 36):
        msg = token_problem(wrong)
        assert msg and "hf_" in msg
        assert wrong not in msg  # never print a key back
    assert "short" in token_problem("hf_abc").lower()


# ------------------------------------------------------- get_hf_token / env
def test_get_hf_token_cleans_env_and_falls_back_to_legacy_name(monkeypatch):
    from scoder.hub import get_hf_token

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert get_hf_token() is None
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", GOOD + "\n")
    assert get_hf_token() == GOOD
    monkeypatch.setenv("HF_TOKEN", "hf_" + "B" * 34 + "\r\n")
    assert get_hf_token() == "hf_" + "B" * 34  # HF_TOKEN wins


# --------------------------------------------------------- error wording
@pytest.mark.parametrize("status,needle", [(401, "not accepted"), (403, "write"), (404, "not found")])
def test_describe_hub_error_tells_the_causes_apart(status, needle):
    from scoder.hub import describe_hub_error

    assert needle in describe_hub_error(HubError(status)).lower()


# ---------------------------------------------------------------- preflight
def test_preflight_sends_the_cleaned_token(fake_hub, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", GOOD + "\r\n")  # what pasting into a Windows/vast terminal can produce
    mod = load_preflight()
    monkeypatch.setattr(sys, "argv", ["hub_preflight.py", "me/repo"])
    assert mod.main() == 0
    assert fake_hub["tokens"] == [GOOD]
    assert "pre-flight OK" in capsys.readouterr().out


def test_preflight_rejects_a_key_from_another_service_before_any_request(fake_hub, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", "cnx_" + "x" * 32)
    monkeypatch.setattr(sys, "argv", ["hub_preflight.py", "me/repo"])
    assert load_preflight().main() == 1
    out = capsys.readouterr().out
    assert "hf_" in out and "cnx_" + "x" * 32 not in out
    assert fake_hub["tokens"] == []  # no network call was made


def test_preflight_401_and_403_give_different_advice(fake_hub, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", GOOD)
    monkeypatch.setattr(sys, "argv", ["hub_preflight.py", "me/repo"])
    mod = load_preflight()

    fake_hub["whoami_error"] = HubError(401)
    assert mod.main() == 1
    assert "not accepted" in capsys.readouterr().out.lower()

    fake_hub["whoami_error"] = None
    fake_hub["write_error"] = HubError(403)
    assert mod.main() == 1
    assert "write" in capsys.readouterr().out.lower()


def test_preflight_requires_a_repo_argument_and_a_token(fake_hub, monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(sys, "argv", ["hub_preflight.py", "me/repo"])
    assert load_preflight().main() == 1
    assert "HF_TOKEN is not set" in capsys.readouterr().out
