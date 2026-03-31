"""Tests for Copilot x-initiator header lifecycle (#3040)."""

import sys
import types
from types import SimpleNamespace

import pytest

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

from run_agent import AIAgent


# ── Helpers ──────────────────────────────────────────────────────────────────

def _tool_defs(*names):
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


class _FakeOpenAI:
    def __init__(self, **kw):
        self.api_key = kw.get("api_key", "test")
        self.base_url = kw.get("base_url", "http://test")
    def close(self):
        pass


def _make_agent(monkeypatch, base_url, api_mode="chat_completions"):
    """Create an AIAgent pointing at the given base_url."""
    monkeypatch.setattr("run_agent.get_tool_definitions", lambda **kw: _tool_defs("web_search"))
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda: {})
    monkeypatch.setattr("run_agent.OpenAI", _FakeOpenAI)
    return AIAgent(
        api_key="test-key",
        base_url=base_url,
        provider="copilot" if "githubcopilot" in base_url else "openrouter",
        api_mode=api_mode,
        max_iterations=4,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


# ── _is_copilot_url tests ───────────────────────────────────────────────────

class TestIsCopilotUrl:
    """_is_copilot_url() detects GitHub Copilot endpoints."""

    def test_standard_copilot_url(self, monkeypatch):
        agent = _make_agent(monkeypatch, "https://api.githubcopilot.com")
        assert agent._is_copilot_url() is True

    def test_copilot_url_with_path(self, monkeypatch):
        agent = _make_agent(monkeypatch, "https://api.githubcopilot.com/v1")
        assert agent._is_copilot_url() is True

    def test_openrouter_url(self, monkeypatch):
        agent = _make_agent(monkeypatch, "https://openrouter.ai/api/v1")
        assert agent._is_copilot_url() is False

    def test_nous_url(self, monkeypatch):
        agent = _make_agent(monkeypatch, "https://inference.nous.hermes")
        assert agent._is_copilot_url() is False

    def test_localhost_url(self, monkeypatch):
        agent = _make_agent(monkeypatch, "http://localhost:8080")
        assert agent._is_copilot_url() is False

    def test_case_insensitive(self, monkeypatch):
        agent = _make_agent(monkeypatch, "https://API.GITHUBCOPILOT.COM")
        assert agent._is_copilot_url() is True

    def test_github_models_url(self, monkeypatch):
        agent = _make_agent(monkeypatch, "https://models.github.ai")
        assert agent._is_copilot_url() is True

    def test_github_models_url_with_path(self, monkeypatch):
        agent = _make_agent(monkeypatch, "https://models.github.ai/v1")
        assert agent._is_copilot_url() is True

    def test_ghe_copilot_url(self, monkeypatch):
        """GitHub Enterprise Copilot endpoints are detected via COPILOT_API_BASE_URL."""
        monkeypatch.setenv("COPILOT_API_BASE_URL", "https://copilot-api.corp.example.com")
        # Reload so is_copilot_url() picks up the new env var
        import importlib, hermes_cli.copilot_auth as _mod
        importlib.reload(_mod)
        agent = _make_agent(monkeypatch, "https://copilot-api.corp.example.com")
        assert agent._is_copilot_url() is True


# ── User-initiated turn flag lifecycle ───────────────────────────────────────

class TestUserInitiatedTurnFlag:
    """_is_user_initiated_turn defaults to False and resets correctly."""

    def test_default_is_false(self, monkeypatch):
        agent = _make_agent(monkeypatch, "https://api.githubcopilot.com")
        assert agent._is_user_initiated_turn is False

    def test_set_true_then_reset_session(self, monkeypatch):
        agent = _make_agent(monkeypatch, "https://api.githubcopilot.com")
        agent._is_user_initiated_turn = True
        agent.reset_session_state()
        assert agent._is_user_initiated_turn is False

    def test_flag_survives_non_reset_operations(self, monkeypatch):
        agent = _make_agent(monkeypatch, "https://api.githubcopilot.com")
        agent._is_user_initiated_turn = True
        # Calling _is_copilot_url() shouldn't touch the flag
        _ = agent._is_copilot_url()
        assert agent._is_user_initiated_turn is True


# ── _build_api_kwargs does NOT inject extra_headers ──────────────────────────

class TestBuildApiKwargsNoExtraHeaders:
    """_build_api_kwargs never includes extra_headers (injected post-preflight)."""

    def test_chat_completions_no_extra_headers(self, monkeypatch):
        agent = _make_agent(monkeypatch, "https://api.githubcopilot.com")
        agent._is_user_initiated_turn = True
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert "extra_headers" not in kwargs

    def test_codex_responses_no_extra_headers(self, monkeypatch):
        agent = _make_agent(
            monkeypatch,
            "https://api.githubcopilot.com",
            api_mode="codex_responses",
        )
        agent._is_user_initiated_turn = True
        # codex_responses builds a different kwargs dict
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert "extra_headers" not in kwargs

    def test_non_copilot_never_gets_extra_headers(self, monkeypatch):
        agent = _make_agent(monkeypatch, "https://openrouter.ai/api/v1")
        agent._is_user_initiated_turn = True
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert "extra_headers" not in kwargs


# ── Preflight does not reject extra_headers when injected correctly ──────────

class TestPreflightExtraHeadersSafe:
    """_preflight_codex_api_kwargs rejects extra_headers (not in allowed_keys)."""

    def test_preflight_rejects_extra_headers_in_kwargs(self, monkeypatch):
        agent = _make_agent(
            monkeypatch,
            "https://api.githubcopilot.com",
            api_mode="codex_responses",
        )
        bad_kwargs = {
            "model": "gpt-5.4",
            "instructions": "test",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            "store": False,
            "extra_headers": {"x-initiator": "user"},
        }
        with pytest.raises(ValueError, match="extra_headers"):
            agent._preflight_codex_api_kwargs(bad_kwargs)

    def test_preflight_accepts_clean_kwargs(self, monkeypatch):
        agent = _make_agent(
            monkeypatch,
            "https://api.githubcopilot.com",
            api_mode="codex_responses",
        )
        clean_kwargs = {
            "model": "gpt-5.4",
            "instructions": "test",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            "store": False,
        }
        result = agent._preflight_codex_api_kwargs(clean_kwargs)
        assert "extra_headers" not in result
        assert result["model"] == "gpt-5.4"


# ── Streaming fallback preserves extra_headers ───────────────────────────────

class TestStreamingFallbackExtraHeaders:
    """Streaming fallback preserves extra_headers across preflight."""

    def test_fallback_preserves_extra_headers(self, monkeypatch):
        agent = _make_agent(
            monkeypatch,
            "https://api.githubcopilot.com",
            api_mode="codex_responses",
        )
        api_kwargs = {
            "model": "gpt-5.4",
            "instructions": "test",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            "store": False,
            "extra_headers": {"x-initiator": "user"},
        }

        captured_kwargs = {}
        fake_response = SimpleNamespace(output=[])

        class FakeResponses:
            def create(self, **kw):
                captured_kwargs.update(kw)
                return fake_response

        fake_client = SimpleNamespace(responses=FakeResponses())

        agent._run_codex_create_stream_fallback(api_kwargs, client=fake_client)

        assert captured_kwargs.get("extra_headers") == {"x-initiator": "user"}
        assert captured_kwargs.get("stream") is True

    def test_fallback_works_without_extra_headers(self, monkeypatch):
        agent = _make_agent(
            monkeypatch,
            "https://api.githubcopilot.com",
            api_mode="codex_responses",
        )
        api_kwargs = {
            "model": "gpt-5.4",
            "instructions": "test",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            "store": False,
        }

        captured_kwargs = {}
        fake_response = SimpleNamespace(output=[])

        class FakeResponses:
            def create(self, **kw):
                captured_kwargs.update(kw)
                return fake_response

        fake_client = SimpleNamespace(responses=FakeResponses())

        agent._run_codex_create_stream_fallback(api_kwargs, client=fake_client)

        assert "extra_headers" not in captured_kwargs
        assert captured_kwargs.get("stream") is True


# ── Flag flip timing: retries must not re-send x-initiator: user ─────────────

class TestFlagFlipOnInjection:
    """Flag flips immediately on injection so retries use 'agent'."""

    def test_flag_false_after_injection(self, monkeypatch):
        agent = _make_agent(monkeypatch, "https://api.githubcopilot.com")
        agent._is_user_initiated_turn = True

        # Simulate the injection block from the main agent loop
        api_kwargs = {}
        if agent._is_copilot_url() and agent._is_user_initiated_turn:
            api_kwargs["extra_headers"] = {"x-initiator": "user"}
            agent._is_user_initiated_turn = False  # same as production code

        assert api_kwargs["extra_headers"] == {"x-initiator": "user"}
        assert agent._is_user_initiated_turn is False

    def test_second_call_has_no_extra_headers(self, monkeypatch):
        agent = _make_agent(monkeypatch, "https://api.githubcopilot.com")
        agent._is_user_initiated_turn = True

        # First iteration — injects and flips
        kwargs1 = {}
        if agent._is_copilot_url() and agent._is_user_initiated_turn:
            kwargs1["extra_headers"] = {"x-initiator": "user"}
            agent._is_user_initiated_turn = False

        # Second iteration (retry or tool follow-up) — should NOT inject
        kwargs2 = {}
        if agent._is_copilot_url() and agent._is_user_initiated_turn:
            kwargs2["extra_headers"] = {"x-initiator": "user"}
            agent._is_user_initiated_turn = False

        assert "extra_headers" in kwargs1
        assert "extra_headers" not in kwargs2

    def test_non_copilot_flag_not_flipped(self, monkeypatch):
        agent = _make_agent(monkeypatch, "https://openrouter.ai/api/v1")
        agent._is_user_initiated_turn = True

        kwargs = {}
        if agent._is_copilot_url() and agent._is_user_initiated_turn:
            kwargs["extra_headers"] = {"x-initiator": "user"}
            agent._is_user_initiated_turn = False

        assert "extra_headers" not in kwargs
        # Flag unchanged — non-Copilot path doesn't touch it
        assert agent._is_user_initiated_turn is True


# ── Integration: copilot_default_headers passes through is_agent_turn ────────

class TestHeaderValues:
    """copilot_default_headers(is_agent_turn=...) sets x-initiator correctly."""

    def test_default_is_agent(self):
        from hermes_cli.models import copilot_default_headers
        assert copilot_default_headers()["x-initiator"] == "agent"

    def test_user_turn(self):
        from hermes_cli.models import copilot_default_headers
        assert copilot_default_headers(is_agent_turn=False)["x-initiator"] == "user"
