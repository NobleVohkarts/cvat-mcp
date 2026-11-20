"""Server-level behaviour.

Tool registration and settings loading happen on import, so per-mode cases run in a fresh
interpreter; the conftest fixture gives each one a clean CVAT_* environment and no .env.
"""

import inspect
import json
import subprocess
import sys

import pytest
from cvat_sdk.api_client.exceptions import ApiException
from mcp.server.mcpserver.exceptions import ToolError

READ_TOOLS = [
    "get_frame_annotations",
    "get_task_overview",
    "list_tasks",
    "render_frame",
    "server_info",
]
LIST_TOOLS = """
import json, cvat_mcp.server as s, cvat_mcp.tools
print(json.dumps({"tools": sorted(s.REGISTERED_TOOLS), "instructions": s.INSTRUCTIONS}))
"""


def run(code: str, monkeypatch, mode: str | None = None, token: str | None = "test"):
    if token is not None:
        monkeypatch.setenv("CVAT_ACCESS_TOKEN", token)
    if mode is not None:
        monkeypatch.setenv("CVAT_MCP_MODE", mode)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)


def registered(monkeypatch, mode):
    return json.loads(run(LIST_TOOLS, monkeypatch, mode).stdout)


@pytest.mark.parametrize("mode", [None, "read-only"])
def test_read_only_mode_registers_only_read_tools(monkeypatch, mode):
    result = registered(monkeypatch, mode)
    assert result["tools"] == READ_TOOLS
    assert "Creating issues is disabled" in result["instructions"]
    assert "create_issue" not in result["instructions"]


def test_write_mode_adds_create_issue(monkeypatch):
    result = registered(monkeypatch, "write")
    assert result["tools"] == sorted(READ_TOOLS + ["create_issue"])
    assert "only when the user asks" in result["instructions"]
    assert "Creating issues is disabled" not in result["instructions"]


def test_bad_mode_fails_at_startup(monkeypatch):
    result = run("from cvat_mcp.__main__ import main; main()", monkeypatch, mode="full")
    assert result.returncode == 1
    assert "CVAT_MCP_MODE" in result.stderr and "Traceback" not in result.stderr


def test_missing_token_fails_at_startup(monkeypatch):
    result = run("from cvat_mcp.__main__ import main; main()", monkeypatch, token=None)
    assert result.returncode == 1
    assert "CVAT_ACCESS_TOKEN" in result.stderr and "Traceback" not in result.stderr


@pytest.fixture
def tool_errors(monkeypatch):
    monkeypatch.setenv("CVAT_ACCESS_TOKEN", "test")  # importing server loads the settings
    from cvat_mcp.server import tool_errors  # pylint: disable=import-outside-toplevel

    return tool_errors


def test_tool_errors_raises_tool_error_and_keeps_signature(tool_errors):
    def get_thing(task_id: int, frame: int = 0) -> str:
        raise ApiException(status=404)

    wrapped = tool_errors(get_thing)
    assert inspect.signature(wrapped) == inspect.signature(get_thing)
    with pytest.raises(ToolError, match="not found"):
        wrapped(1)


def test_tool_errors_lets_unexpected_exceptions_through(tool_errors):
    def broken():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        tool_errors(broken)()
