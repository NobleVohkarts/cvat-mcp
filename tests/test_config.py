from pathlib import Path

import pytest
from pydantic import ValidationError

from cvat_mcp.config import Settings, resolve_in_workdir
from cvat_mcp.errors import InvalidArgument


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


def test_path_inside_workdir(tmp_path):
    assert resolve_in_workdir(tmp_path, "a/b.zip") == tmp_path.resolve() / "a" / "b.zip"


@pytest.mark.parametrize("path", ["../x", "/etc/passwd", "a/../../x", "", ".", "out\x00.zip"])
def test_path_escaping_workdir(tmp_path, path):
    with pytest.raises(InvalidArgument, match="CVAT_MCP_WORKDIR"):
        resolve_in_workdir(tmp_path, path)


def test_symlink_out_of_workdir(tmp_path):
    (tmp_path / "link").symlink_to("/tmp")
    with pytest.raises(InvalidArgument, match="CVAT_MCP_WORKDIR"):
        resolve_in_workdir(tmp_path, "link/x")
