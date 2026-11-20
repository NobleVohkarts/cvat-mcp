"""Server-level behaviour. Each case runs in a fresh interpreter: tool registration and settings
loading happen on import, so a new process is the honest way to observe them per mode."""

import os
import subprocess
import sys
import textwrap


def run(code: str, tmp_path, mode: str | None = None, token: str | None = "test"):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CVAT_")}
    if token is not None:
        env["CVAT_ACCESS_TOKEN"] = token
    if mode is not None:
        env["CVAT_MCP_MODE"] = mode
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=tmp_path,  # no .env here
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_tool_errors_raises_tool_error_and_keeps_signature(tmp_path):
    result = run(
        """
        import inspect
        from cvat_sdk.api_client.exceptions import ApiException
        from mcp.server.mcpserver.exceptions import ToolError
        from cvat_mcp.server import tool_errors

        def get_thing(task_id: int, frame: int = 0) -> str:
            raise ApiException(status=404)

        wrapped = tool_errors(get_thing)
        assert inspect.signature(wrapped) == inspect.signature(get_thing)
        try:
            wrapped(1)
        except ToolError as e:
            assert "not found" in str(e)
        else:
            raise AssertionError("no ToolError")
        """,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr


def test_tool_errors_lets_unexpected_exceptions_through(tmp_path):
    result = run(
        """
        from cvat_mcp.server import tool_errors

        def broken():
            raise RuntimeError("boom")

        try:
            tool_errors(broken)()
        except RuntimeError:
            pass
        """,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr


READ_TOOLS = [
    "get_frame_annotations",
    "get_task_overview",
    "list_tasks",
    "render_frame",
    "server_info",
]
LIST_TOOLS = """
import cvat_mcp.server as s, cvat_mcp.tools
print(sorted(s.REGISTERED_TOOLS))
print(s.INSTRUCTIONS)
"""


def test_read_only_mode_registers_only_read_tools(tmp_path):
    result = run(LIST_TOOLS, tmp_path, mode="read-only")
    tools, _, instructions = result.stdout.partition("\n")
    assert tools == str(READ_TOOLS)
    assert "Creating issues is disabled" in instructions
    assert "report the problems to the user" in instructions


def test_write_mode_adds_create_issue(tmp_path):
    result = run(LIST_TOOLS, tmp_path, mode="write")
    tools, _, instructions = result.stdout.partition("\n")
    assert tools == str(sorted(READ_TOOLS + ["create_issue"]))
    assert "create_issue, one per problem" in instructions
    assert "Creating issues is disabled" not in instructions


def test_default_mode_is_read_only(tmp_path):
    tools = run(LIST_TOOLS, tmp_path).stdout.partition("\n")[0]
    assert tools == str(READ_TOOLS)


def test_bad_mode_fails_at_startup(tmp_path):
    result = run("from cvat_mcp.__main__ import main; main()", tmp_path, mode="full")
    assert result.returncode == 1
    assert "CVAT_MCP_MODE" in result.stderr and "Traceback" not in result.stderr


def test_missing_token_fails_at_startup(tmp_path):
    result = run("from cvat_mcp.__main__ import main; main()", tmp_path, token=None)
    assert result.returncode == 1
    assert "CVAT_ACCESS_TOKEN" in result.stderr and "Traceback" not in result.stderr
