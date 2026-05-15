"""Tests for GitHub Enterprise Copilot auth support.

Covers the env-var-overrideable constants and helpers added for GHE:
  - COPILOT_AUTH_MODE=oauth bypass
  - COPILOT_DEVICE_CODE_URL / COPILOT_ACCESS_TOKEN_URL overrides
  - COPILOT_TOKEN_EXCHANGE_URL override
  - COPILOT_GH_HOST --hostname passthrough (with clean_env)
  - is_copilot_url() with custom COPILOT_API_BASE_URL
  - is_classic_pat() helper
"""
from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reload_copilot_auth(monkeypatch, env: dict):
    """Re-import copilot_auth with a controlled environment."""
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    # Remove cached module so os.getenv() at module level is re-evaluated
    sys.modules.pop("hermes_cli.copilot_auth", None)
    import hermes_cli.copilot_auth as mod
    return mod


# ---------------------------------------------------------------------------
# resolve_copilot_token — COPILOT_AUTH_MODE=oauth
# ---------------------------------------------------------------------------

class TestCopilotAuthMode:
    def test_oauth_mode_returns_empty(self, monkeypatch):
        """COPILOT_AUTH_MODE=oauth must bypass env-var and gh CLI lookup."""
        monkeypatch.setenv("COPILOT_AUTH_MODE", "oauth")
        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gho_shouldnotbeused")
        import hermes_cli.copilot_auth as mod
        token, source = mod.resolve_copilot_token()
        assert token == ""
        assert source == ""

    def test_oauth_mode_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("COPILOT_AUTH_MODE", "OAUTH")
        import hermes_cli.copilot_auth as mod
        token, _ = mod.resolve_copilot_token()
        assert token == ""

    def test_unset_mode_uses_env_vars(self, monkeypatch):
        monkeypatch.delenv("COPILOT_AUTH_MODE", raising=False)
        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gho_validtoken")
        import hermes_cli.copilot_auth as mod
        token, source = mod.resolve_copilot_token()
        assert token == "gho_validtoken"
        assert source == "COPILOT_GITHUB_TOKEN"


# ---------------------------------------------------------------------------
# COPILOT_DEVICE_CODE_URL / COPILOT_ACCESS_TOKEN_URL at module level
# ---------------------------------------------------------------------------

class TestOverrideableOAuthURLs:
    def test_defaults_to_github_com(self, monkeypatch):
        monkeypatch.delenv("COPILOT_DEVICE_CODE_URL", raising=False)
        monkeypatch.delenv("COPILOT_ACCESS_TOKEN_URL", raising=False)
        mod = _reload_copilot_auth(monkeypatch, {})
        assert mod.COPILOT_DEVICE_CODE_URL == "https://github.com/login/device/code"
        assert mod.COPILOT_ACCESS_TOKEN_URL == "https://github.com/login/oauth/access_token"

    def test_ghe_override(self, monkeypatch):
        mod = _reload_copilot_auth(monkeypatch, {
            "COPILOT_DEVICE_CODE_URL": "https://cancomictdev.ghe.com/login/device/code",
            "COPILOT_ACCESS_TOKEN_URL": "https://cancomictdev.ghe.com/login/oauth/access_token",
        })
        assert "cancomictdev.ghe.com" in mod.COPILOT_DEVICE_CODE_URL
        assert "cancomictdev.ghe.com" in mod.COPILOT_ACCESS_TOKEN_URL


# ---------------------------------------------------------------------------
# COPILOT_TOKEN_EXCHANGE_URL at module level
# ---------------------------------------------------------------------------

class TestTokenExchangeURL:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("COPILOT_TOKEN_EXCHANGE_URL", raising=False)
        mod = _reload_copilot_auth(monkeypatch, {})
        assert mod._TOKEN_EXCHANGE_URL == "https://api.github.com/copilot_internal/v2/token"

    def test_ghe_override(self, monkeypatch):
        mod = _reload_copilot_auth(monkeypatch, {
            "COPILOT_TOKEN_EXCHANGE_URL": "https://api.cancomictdev.ghe.com/copilot_internal/v2/token",
        })
        assert "cancomictdev.ghe.com" in mod._TOKEN_EXCHANGE_URL


# ---------------------------------------------------------------------------
# is_copilot_url()
# ---------------------------------------------------------------------------

class TestIsCopilotUrl:
    def test_cloud_url(self, monkeypatch):
        monkeypatch.delenv("COPILOT_API_BASE_URL", raising=False)
        import hermes_cli.copilot_auth as mod
        assert mod.is_copilot_url("https://api.githubcopilot.com/v1")
        assert mod.is_copilot_url("https://models.github.ai/inference")

    def test_unrelated_url(self, monkeypatch):
        monkeypatch.delenv("COPILOT_API_BASE_URL", raising=False)
        import hermes_cli.copilot_auth as mod
        assert not mod.is_copilot_url("https://api.openai.com/v1")

    def test_custom_ghe_base_url(self, monkeypatch):
        monkeypatch.setenv("COPILOT_API_BASE_URL", "https://cancomictdev.ghe.com/v1")
        import hermes_cli.copilot_auth as mod
        assert mod.is_copilot_url("https://cancomictdev.ghe.com/v1/chat/completions")

    def test_empty_url(self, monkeypatch):
        monkeypatch.delenv("COPILOT_API_BASE_URL", raising=False)
        import hermes_cli.copilot_auth as mod
        assert not mod.is_copilot_url("")
        assert not mod.is_copilot_url("")  # None-equivalent: pass empty string


# ---------------------------------------------------------------------------
# is_classic_pat()
# ---------------------------------------------------------------------------

class TestIsClassicPat:
    def test_ghp_prefix_is_classic(self):
        import hermes_cli.copilot_auth as mod
        assert mod.is_classic_pat("ghp_sometoken")

    def test_gho_is_not_classic(self):
        import hermes_cli.copilot_auth as mod
        assert not mod.is_classic_pat("gho_validtoken")

    def test_fine_grained_pat_is_not_classic(self):
        import hermes_cli.copilot_auth as mod
        assert not mod.is_classic_pat("github_pat_validtoken")


# ---------------------------------------------------------------------------
# _try_gh_cli_token — COPILOT_GH_HOST passes --hostname
# ---------------------------------------------------------------------------

class TestGHCLITokenResolution:
    def test_copilot_gh_host_passes_hostname(self, monkeypatch, tmp_path):
        """When COPILOT_GH_HOST is set, gh must be called with --hostname."""
        monkeypatch.setenv("COPILOT_GH_HOST", "cancomictdev.ghe.com")
        # Ensure clean_env strips GH_TOKEN / GITHUB_TOKEN
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_shouldnotleak")
        monkeypatch.setenv("GH_TOKEN", "ghp_shouldnotleak2")

        import hermes_cli.copilot_auth as mod

        captured_cmds = []
        captured_envs = []

        def fake_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            captured_envs.append(kwargs.get("env", {}))
            result = MagicMock()
            result.returncode = 0
            result.stdout = "gho_fakeGHEtoken\n"
            return result

        with patch.object(mod, "_gh_cli_candidates", return_value=["/usr/bin/gh"]):
            with patch("subprocess.run", side_effect=fake_run):
                token = mod._try_gh_cli_token()

        assert token == "gho_fakeGHEtoken"
        assert "--hostname" in captured_cmds[0]
        assert "cancomictdev.ghe.com" in captured_cmds[0]
        # clean_env must not contain GITHUB_TOKEN or GH_TOKEN
        assert "GITHUB_TOKEN" not in captured_envs[0]
        assert "GH_TOKEN" not in captured_envs[0]

    def test_no_host_no_hostname_flag(self, monkeypatch):
        monkeypatch.delenv("COPILOT_GH_HOST", raising=False)
        import hermes_cli.copilot_auth as mod

        captured = []

        def fake_run(cmd, **kwargs):
            captured.append(cmd)
            result = MagicMock()
            result.returncode = 0
            result.stdout = "gho_cloudtoken\n"
            return result

        with patch.object(mod, "_gh_cli_candidates", return_value=["/usr/bin/gh"]):
            with patch("subprocess.run", side_effect=fake_run):
                token = mod._try_gh_cli_token()

        assert token == "gho_cloudtoken"
        assert "--hostname" not in captured[0]
