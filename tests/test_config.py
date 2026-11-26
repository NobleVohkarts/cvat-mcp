import pytest
from pydantic import ValidationError

from cvat_mcp.config import Settings


def test_defaults(monkeypatch):
    monkeypatch.setenv("CVAT_ACCESS_TOKEN", "secret-value")
    s = Settings()
    assert s.url == "http://localhost:8080"
    assert s.mcp_mode == "read-only"


def test_missing_token_raises():
    with pytest.raises(ValidationError, match="access_token"):
        Settings()


def test_token_hidden(monkeypatch):
    monkeypatch.setenv("CVAT_ACCESS_TOKEN", "secret-value")
    s = Settings()
    assert "secret-value" not in repr(s)
    assert "secret-value" not in str(s.model_dump())
    assert s.access_token.get_secret_value() == "secret-value"


def test_bad_mode_rejected(monkeypatch):
    monkeypatch.setenv("CVAT_ACCESS_TOKEN", "secret-value")
    monkeypatch.setenv("CVAT_MCP_MODE", "full")
    with pytest.raises(ValidationError):
        Settings()


def test_env_file_is_read(tmp_path):
    (tmp_path / ".env").write_text("CVAT_ACCESS_TOKEN=from-file\nCVAT_MCP_MODE=write\n")
    s = Settings()
    assert s.access_token.get_secret_value() == "from-file"
    assert s.mcp_mode == "write"


def test_old_workdir_setting_is_ignored(tmp_path):
    (tmp_path / ".env").write_text("CVAT_ACCESS_TOKEN=t\nCVAT_MCP_WORKDIR=workdir\n")
    s = Settings()  # an .env from before the server dropped local files still loads
    assert not hasattr(s, "mcp_workdir")


def test_http_mode_defaults(monkeypatch):
    monkeypatch.setenv("CVAT_ACCESS_TOKEN", "t")
    s = Settings()
    assert s.mcp_transport == "stdio" and s.mcp_port == 8765
    assert s.redirect_hosts == {"claude.ai", "claude.com", "localhost", "127.0.0.1"}


@pytest.mark.parametrize("url", [None, "http://example.com", "https://x.trycloudflare.com/mcp"])
def test_http_mode_needs_https_public_url(monkeypatch, url):
    monkeypatch.setenv("CVAT_ACCESS_TOKEN", "t")
    monkeypatch.setenv("CVAT_MCP_TRANSPORT", "http")
    if url:
        monkeypatch.setenv("CVAT_MCP_PUBLIC_URL", url)
    with pytest.raises(ValidationError) as error:
        Settings()
    assert error.value.errors()[0]["loc"] == ("mcp_public_url",)


@pytest.mark.parametrize("url", ["https://x.trycloudflare.com", "http://localhost:8765"])
def test_http_mode_accepts_https_or_local(monkeypatch, url):
    monkeypatch.setenv("CVAT_ACCESS_TOKEN", "t")
    monkeypatch.setenv("CVAT_MCP_TRANSPORT", "http")
    monkeypatch.setenv("CVAT_MCP_PUBLIC_URL", url)
    assert Settings().public_url == url
