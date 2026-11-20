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
