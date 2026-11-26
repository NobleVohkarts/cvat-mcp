"""Server-level behaviour.

Tool registration and settings loading happen on import, so per-mode cases run in a fresh
interpreter; the conftest fixture gives each one a clean CVAT_* environment and no .env.
"""

import inspect
import json
import subprocess
import sys
from importlib.util import find_spec

import pytest
from cvat_sdk.api_client.exceptions import ApiException
from mcp.server.mcpserver.exceptions import ToolError

import cvat_mcp

# The auto-annotation tools exist only with the extra installed (torch, torchvision, skimage).
AUTO_ANNOTATION = all(find_spec(m) is not None for m in ("torch", "torchvision", "skimage"))
AA_TOOLS = ["auto_annotate", "get_auto_annotation_run"] if AUTO_ANNOTATION else []
AA_WRITE = ["auto_annotate"] if AUTO_ANNOTATION else []  # destructive and open world
WITHOUT_EXTRA = """
import importlib.util
real = importlib.util.find_spec
hidden = ("torch", "torchvision", "skimage")
importlib.util.find_spec = lambda name, package=None: None if name in hidden else real(name, package)
"""
READ_TOOLS = [
    "create_report",
    "download_annotations",
    "get_frame_annotations",
    "get_task_overview",
    "list_cloud_storages",
    "list_issues",
    "list_tasks",
    "render_frame",
    "server_info",
]
LIST_TOOLS = """
import anyio, json, cvat_mcp.server as s, cvat_mcp.tools
tools = anyio.run(s.mcp.list_tools)
destructive = [t.name for t in tools if t.annotations.destructive_hint]
open_world = [t.name for t in tools if t.annotations.open_world_hint]
with_job_id = [t.name for t in tools if "job_id" in t.input_schema.get("properties", {})]
print(json.dumps({
    "tools": sorted(s.REGISTERED_TOOLS), "instructions": s.INSTRUCTIONS, "destructive": destructive,
    "open_world": open_world, "with_job_id": with_job_id,
    "descriptions": " ".join(t.description or "" for t in tools),
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
    assert "download_annotations for all of a job's annotations" in result["instructions"]
    assert "Write tools are disabled: the server runs in read-only mode" in result["instructions"]
    assert "create_issue" not in result["instructions"]
    assert "Issues belong" not in result["instructions"]
    assert "delete_issue" not in result["instructions"]
    assert "update_annotations" not in result["instructions"]
    assert "create_cloud_storage" not in result["instructions"]
    assert "import_annotations" not in result["instructions"]
    assert "is not available through this server" not in result["instructions"]
    assert "call create_report, then\nrender_frame the conflict frames" in result["instructions"]
    assert "job_id of the ground-truth job to see the ground truth" in result["instructions"]


def test_write_mode_adds_write_tools(monkeypatch):
    result = registered(monkeypatch, "write")
    write_tools = [
        "create_annotations",
        "create_cloud_storage",
        "create_gt_job",
        "create_issue",
        "create_task",
        "delete_annotations",
        "delete_cloud_storage",
        "delete_gt_job",
        "delete_issue",
        "delete_task",
        "export_dataset",
        "import_annotations",
        "update_annotations",
        "update_cloud_storage",
        "update_job",
        "update_quality_settings",
        "update_task",
    ]
    assert result["tools"] == sorted(READ_TOOLS + write_tools + AA_TOOLS)
    assert "only when the user asks" in result["instructions"]
    assert "Issues belong to the job" in result["instructions"]
    assert "create_issue the new ones, then delete_issue the old ones" in result["instructions"]
    assert "only when the user asks to fix" in result["instructions"]
    assert sorted(result["destructive"]) == AA_WRITE + [  # replace clears the run's frames
        "delete_annotations",
        "delete_cloud_storage",
        "delete_gt_job",
        "delete_issue",
        "delete_task",
        "export_dataset",  # overwrite=true replaces an object in the bucket
        "import_annotations",  # mode="replace" deletes the target's annotations first
    ]
    assert "update_job" in result["instructions"]
    assert "update_quality_settings" in result["instructions"]
    assert "ground-truth jobs of tasks owned by" in result["instructions"]
    assert "call create_report, then\nrender_frame the conflict frames" in result["instructions"]
    assert "job_id of the ground-truth job to see the ground truth" in result["instructions"]
    assert "edit ground truth only when the user explicitly asks" in result["instructions"]
    annotation_tools = {
        "create_annotations",
        "delete_annotations",
        "download_annotations",
        "get_frame_annotations",
        "render_frame",
        "update_annotations",
    }
    assert annotation_tools <= set(result["with_job_id"])
    assert "create_task" in result["instructions"]
    assert "only tasks owned by" in result["instructions"]
    # urls and cloud storages make CVAT fetch from any host
    assert sorted(result["open_world"]) == AA_WRITE + [  # first use downloads model weights
        "create_cloud_storage",
        "create_task",
        "export_dataset",  # CVAT writes into the bucket
        "import_annotations",  # CVAT reads the file from the bucket
        "update_cloud_storage",  # CVAT re-checks the bucket on every update
    ]
    assert "import_annotations" in result["instructions"]
    assert "appends unless" in result["instructions"]
    assert "export_dataset" in result["instructions"]
    assert "ask the user before overwrite" in result["instructions"]
    assert "keys go only to create_cloud_storage and update_cloud_storage" in result["instructions"]
    assert "never repeat keys" in result["instructions"]
    assert "list_cloud_storages for the bucket's id" in result["instructions"]
    assert "disabled" not in result["instructions"]
    assert ("get_auto_annotation_run" in result["instructions"]) == AUTO_ANNOTATION
    for text in (result["instructions"], result["descriptions"]):  # no local files any more
        assert "work directory" not in text and "WORKDIR" not in text


def test_write_mode_without_the_extra_has_no_auto_annotation(monkeypatch):
    result = run(WITHOUT_EXTRA + LIST_TOOLS, monkeypatch, "write")
    assert result.returncode == 0, result.stderr
    listed = json.loads(result.stdout)
    assert not {"auto_annotate", "get_auto_annotation_run"} & set(listed["tools"])
    assert "auto_annotate" not in listed["instructions"]


@pytest.mark.skipif(not AUTO_ANNOTATION, reason="needs the auto-annotation extra")
def test_read_only_mode_has_no_auto_annotation(monkeypatch):
    result = registered(monkeypatch, "read-only")
    assert "auto_annotate" not in result["tools"]
    assert "auto_annotate" not in result["instructions"]


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


WRITE_IN_PROCESS = """
import anyio, json
import cvat_mcp.server as s, cvat_mcp.tools
from mcp import Client
from mcp.shared.exceptions import MCPError
from cvat_mcp.errors import InvalidArgument

class Reached:
    def close(self):
        pass

    def __getattr__(self, name):  # the service's first use of the SDK client
        raise InvalidArgument("reached the service")

s._connect = lambda _settings: (Reached(), "admin")
CALLS = {
    "create_task": {"name": "t", "labels": [{"name": "car"}], "urls": ["https://e.com/a.jpg"]},
    "import_annotations": {"task_id": 1, "cloud_storage_id": 1, "cloud_path": "a.json"},
    "export_dataset": {"task_id": 1, "format": "COCO 1.0", "cloud_storage_id": 1,
                       "cloud_path": "a.zip"},
    "old_files": ["create_task", {"name": "t", "labels": [{"name": "car"}], "files": ["a.jpg"]}],
    "old_file": ["import_annotations", {"task_id": 1, "file": "a.json"}],
}

async def outcome(client, name, arguments):
    try:
        result = await client.call_tool(name, arguments)
        return ["result", result.content[0].text]
    except MCPError as error:
        return ["error", error.error.message]

async def main():
    async with Client(s.mcp) as client:
        out = {}
        for key, call in CALLS.items():
            name, arguments = call if isinstance(call, list) else (key, call)
            out[key] = await outcome(client, name, arguments)
        print(json.dumps(out))

anyio.run(main)
"""


def test_write_tools_pass_their_arguments_to_the_services(monkeypatch):
    result = run(WRITE_IN_PROCESS, monkeypatch, "write")
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout.strip().splitlines()[-1])
    for name in ("create_task", "import_annotations", "export_dataset"):
        assert out[name][1].endswith(": reached the service"), (name, out[name])
    # local files are gone: the old arguments are unknown, not silently dropped
    assert out["old_files"][0] == "error" and "files" in out["old_files"][1]
    assert out["old_file"][0] == "error" and "file" in out["old_file"][1]
