from pathlib import Path

import pytest
from pydantic import ValidationError

from cvat_mcp.config import Settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for name in ("CVAT_URL", "CVAT_ACCESS_TOKEN", "CVAT_MCP_MODE", "CVAT_MCP_WORKDIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)  # no real .env is read


def test_defaults(monkeypatch):
    monkeypatch.setenv("CVAT_ACCESS_TOKEN", "secret-value")
    s = Settings()
    assert s.url == "http://localhost:8080"
    assert s.mcp_mode == "read-only"
    assert s.mcp_workdir == Path("workdir")


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
