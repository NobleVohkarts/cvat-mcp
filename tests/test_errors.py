import inspect

import pytest
from cvat_sdk.api_client.exceptions import ApiException
from mcp.server.mcpserver.exceptions import ToolError

from cvat_mcp.errors import explain, tool_errors


def _api(status, reason=None, body=None):
    exc = ApiException(status=status, reason=reason)
    exc.body = body
    return exc


def test_not_found_points_to_listing_tools():
    message = explain(_api(404))
    assert "not found" in message and "list_tasks" in message


def test_forbidden_mentions_read_only_token():
    message = explain(_api(403))
    assert "permission" in message and "read-only" in message


def test_unauthorized_mentions_token():
    assert "token" in explain(_api(401))


def test_bad_request_includes_server_reason():
    message = explain(_api(400, "Bad Request", '{"frame": ["bad"]}'))
    assert "rejected" in message and "frame" in message


def test_own_validation_message_passes_through():
    assert explain(ValueError("frame 99 is outside 0..39")) == "frame 99 is outside 0..39"


def test_unexpected_exception_is_not_explained():
    assert explain(RuntimeError("boom")) is None


def test_tool_errors_raises_tool_error_and_keeps_signature():
    def get_thing(task_id: int, frame: int = 0) -> str:
        raise _api(404)

    wrapped = tool_errors(get_thing)
    assert inspect.signature(wrapped) == inspect.signature(get_thing)
    with pytest.raises(ToolError, match="not found"):
        wrapped(1)


def test_tool_errors_lets_unexpected_exceptions_through():
    def broken():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        tool_errors(broken)()
