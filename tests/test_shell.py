"""The agent shell runs model-written commands, so it must not leak secrets or run obvious disasters."""
import pytest

shell = pytest.importorskip("agent.shell", reason="needs bash")


def test_blocklist_stops_obviously_destructive_commands(tmp_path):
    for cmd in ("rm -rf /", "rm -rf ~", "sudo ls", "mkfs.ext4 /dev/sda", "shutdown now", ":(){ :|:& };:"):
        code, out = shell.run(cmd, str(tmp_path))
        assert code == 1 and "blocked" in out.lower(), cmd


def test_runs_a_command_in_the_workspace(tmp_path):
    (tmp_path / "f.txt").write_text("hello")
    code, out = shell.run("cat f.txt", str(tmp_path))
    assert code == 0 and out.strip() == "hello"


def test_timeout_is_reported(tmp_path):
    code, out = shell.run("sleep 5", str(tmp_path), timeout=1)
    assert code == -1 and "timed out" in out


def test_secrets_in_the_environment_are_not_visible_to_agent_commands(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_DUMMYSECRET_FOR_TEST_ONLY")
    monkeypatch.setenv("SOME_SERVICE_API_KEY", "dummy-api-key-value")
    monkeypatch.setenv("HARMLESS_SETTING", "visible")
    _, out = shell.run("env", str(tmp_path))
    assert "hf_DUMMYSECRET_FOR_TEST_ONLY" not in out
    assert "dummy-api-key-value" not in out
    assert "HARMLESS_SETTING=visible" in out
