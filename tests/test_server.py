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

import cvat_mcp

READ_TOOLS = [
    "export_dataset",
    "get_frame_annotations",
    "get_task_overview",
    "list_issues",
    "list_tasks",
    "render_frame",
    "server_info",
]
LIST_TOOLS = """
import anyio, json, cvat_mcp.server as s, cvat_mcp.tools
destructive = [t.name for t in anyio.run(s.mcp.list_tools) if t.annotations.destructive_hint]
print(json.dumps({
    "tools": sorted(s.REGISTERED_TOOLS), "instructions": s.INSTRUCTIONS, "destructive": destructive
}))
"""


def run(code: str, monkeypatch, mode: str | None = None, token: str | None = "test"):
    if token is not None:
        monkeypatch.setenv("CVAT_ACCESS_TOKEN", token)
    if mode is not None:
        monkeypatch.setenv("CVAT_MCP_MODE", mode)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=60,
    )


def registered(monkeypatch, mode):
    result = run(LIST_TOOLS, monkeypatch, mode)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("mode", [None, "read-only"])
def test_read_only_mode_registers_only_read_tools(monkeypatch, mode):
    result = registered(monkeypatch, mode)
    assert result["tools"] == READ_TOOLS
    assert "Creating issues is disabled" in result["instructions"]
    assert "create_issue" not in result["instructions"]
    assert "Issues belong" not in result["instructions"]
    assert "delete_issue" not in result["instructions"]


def test_write_mode_adds_issue_tools(monkeypatch):
    result = registered(monkeypatch, "write")
    assert result["tools"] == sorted(READ_TOOLS + ["create_issue", "delete_issue"])
    assert "only when the user asks" in result["instructions"]
    assert "Issues belong to the job" in result["instructions"]
    assert "create_issue the new ones, then delete_issue the old ones" in result["instructions"]
    assert result["destructive"] == ["delete_issue"]
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


HTTP_CHECKS = """
import json
from starlette.testclient import TestClient
import cvat_mcp.server as s, cvat_mcp.tools
app = s.mcp.streamable_http_app(transport_security=s.transport_security())
client = TestClient(app, base_url="http://localhost:8765", follow_redirects=False)  # no lifespan
mcp = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
print(json.dumps({
    "as": client.get("/.well-known/oauth-authorization-server").json(),
    "prm": client.get("/.well-known/oauth-protected-resource/mcp").status_code,
    "mcp": mcp.status_code,
    "www": mcp.headers.get("www-authenticate", ""),
    "login": client.get("/login?request=nope").status_code,
}))
"""


def test_http_mode_serves_oauth_and_protects_mcp(monkeypatch):
    monkeypatch.setenv("CVAT_MCP_TRANSPORT", "http")
    monkeypatch.setenv("CVAT_MCP_PUBLIC_URL", "http://localhost:8765")
    result = run(HTTP_CHECKS, monkeypatch)
    assert result.returncode == 0, result.stderr
    checks = json.loads(result.stdout)
    assert checks["as"]["registration_endpoint"] == "http://localhost:8765/register"
    assert checks["prm"] == 200
    assert checks["mcp"] == 401 and "resource_metadata" in checks["www"]
    assert checks["login"] == 400


def test_stdio_mode_has_no_oauth_routes(monkeypatch):
    result = run(
        "import json, cvat_mcp.server as s, cvat_mcp.tools\n"
        "print(json.dumps([r.path for r in s.mcp.streamable_http_app().routes]))",
        monkeypatch,
    )
    assert result.returncode == 0, result.stderr
    paths = json.loads(result.stdout)
    assert "/login" not in paths and not any(p.startswith("/.well-known") for p in paths)


def test_http_mode_without_public_url_fails_at_startup(monkeypatch):
    monkeypatch.setenv("CVAT_MCP_TRANSPORT", "http")
    result = run("from cvat_mcp.__main__ import main; main()", monkeypatch)
    assert result.returncode == 1
    assert "CVAT_MCP_PUBLIC_URL" in result.stderr and "Traceback" not in result.stderr


def test_lifespan_sets_the_login_owner_from_the_token(monkeypatch):
    monkeypatch.setenv("CVAT_MCP_TRANSPORT", "http")
    monkeypatch.setenv("CVAT_MCP_PUBLIC_URL", "http://localhost:8765")
    code = """
import anyio
from types import SimpleNamespace as Obj
import cvat_mcp.server as s
s._connect = lambda _settings: (Obj(close=lambda: None), "admin")

async def main():
    async with s.lifespan(s.mcp):
        print(s.oauth.owner)

anyio.run(main)
"""
    result = run(code, monkeypatch)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "admin"


def test_http_mode_reports_unreachable_cvat_in_one_line(monkeypatch):
    monkeypatch.setenv("CVAT_MCP_TRANSPORT", "http")
    monkeypatch.setenv("CVAT_MCP_PUBLIC_URL", "http://localhost:8799")
    monkeypatch.setenv("CVAT_MCP_PORT", "8799")
    monkeypatch.setenv("CVAT_URL", "http://localhost:9")
    result = run("from cvat_mcp.__main__ import main; main()", monkeypatch)
    assert result.returncode == 1
    assert "cannot reach CVAT" in result.stderr and "Traceback" not in result.stderr


IN_PROCESS = """
import anyio, json
from types import SimpleNamespace as Obj
import cvat_mcp.server as s, cvat_mcp.tools
from mcp import Client
from mcp.shared.exceptions import MCPError
s._connect = lambda _settings: (Obj(close=lambda: None), "admin")

async def outcome(client, name, arguments):
    try:
        result = await client.call_tool(name, arguments)
        return ["result", result.is_error, result.content[0].text]
    except MCPError as error:
        return ["error", error.error.code, error.error.message]

async def passed_on(ctx):
    return "passed on to mcp"

async def main():
    raw = Obj(method="tools/call", params={"name": "render_frame", "arguments": [{"x": 1}]})
    not_object = await s.reject_unknown_arguments(raw, passed_on)
    async with Client(s.mcp) as client:
        print(json.dumps({
            "version": client.server_info.version,
            "not_object": not_object,
            "unknown": await outcome(client, "render_frame", {"task_id": 1, "frame": 0, "clean": 1}),
            "no_args": await outcome(client, "server_info", {"verbose": True}),
            "known": await outcome(client, "get_frame_annotations", {"task_id": 1, "frame": 0}),
        }))

anyio.run(main)
"""


def test_strict_arguments_and_version_in_process(monkeypatch):
    result = run(IN_PROCESS, monkeypatch)
    assert result.returncode == 0, result.stderr
    checks = json.loads(result.stdout.strip().splitlines()[-1])
    assert checks["version"] == cvat_mcp.__version__  # the MCP handshake reports the build
    kind, code, message = checks["unknown"]
    assert (kind, code) == ("error", -32602)  # JSON-RPC invalid params, shaped by mcp's runner
    assert "unknown argument(s) clean for render_frame" in message
    assert "include_clean" in message and "reconnect" in message
    assert "takes no arguments" in checks["no_args"][2]
    assert checks["not_object"] == "passed on to mcp"  # mcp's own validation answers
    # known arguments reach the tool body (which fails on the fake client)
    assert checks["known"] == ["result", True, "Error executing tool get_frame_annotations"]
