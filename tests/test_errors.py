import inspect

import pytest
from cvat_sdk.api_client.exceptions import ApiException
from mcp.server.mcpserver.exceptions import ToolError

from cvat_mcp.errors import InvalidArgument, explain, tool_errors
from cvat_mcp.services.annotations import FrameObject


def _api(status, reason=None, body=None):
    exc = ApiException(status=status, reason=reason)
    exc.body = body
    return exc


@pytest.mark.parametrize(
    "status, body, words",
    [
        (404, None, ["not found", "list_tasks"]),
        (403, None, ["permission", "read-only"]),
        (401, None, ["token"]),
        (400, '{"frame": ["bad"]}', ["rejected", "frame"]),
    ],
)
def test_api_errors(status, body, words):
    message = explain(_api(status, "Reason", body))
    assert all(w in message for w in words)


def test_bad_request_body_bytes_are_decoded():
    message = explain(_api(400, "Bad Request", b'{"position": ["bad"]}'))
    assert "b'" not in message and '"position"' in message


def test_own_validation_message_passes_through():
    assert explain(InvalidArgument("frame 99 is outside 0..39")) == "frame 99 is outside 0..39"


def test_other_value_errors_are_unexpected():
    try:
        FrameObject.model_validate({})
    except ValueError as exc:  # pydantic's ValidationError is a ValueError too
        assert explain(exc) is None
    assert explain(ValueError("internal")) is None


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
