"""Turn exceptions into short messages the model can act on.

mcp reports any exception other than ToolError as a bare "Error executing tool <name>", so every
tool is wrapped with tool_errors() and expected failures are re-raised as ToolError with a hint.
"""

import functools
from collections.abc import Callable
from typing import TypeVar

from cvat_sdk.api_client.exceptions import ApiException
from mcp.server.mcpserver.exceptions import ToolError

F = TypeVar("F", bound=Callable)


class InvalidArgument(ValueError):
    """The model passed something wrong; the message says what to pass instead."""


_HINTS = {
    401: "CVAT rejected the access token (expired or revoked): create a new Personal Access "
    "Token in the CVAT UI and restart the server.",
    403: "CVAT denied permission for this action. The current user may lack access to this "
    "resource, or the access token is read-only (write actions need a token without 'read only').",
    404: "CVAT: not found. The task, job or frame id does not exist or is not visible to this user. "
    "Use list_tasks and get_task_overview to find valid ids.",
}


def explain(exc: Exception) -> str | None:
    """A message for the model, or None for an unexpected error."""
    if isinstance(exc, InvalidArgument):
        return str(exc)
    if isinstance(exc, ApiException):
        if exc.status in _HINTS:
            return _HINTS[exc.status]
        if exc.status == 400:
            body = exc.body.decode(errors="replace") if isinstance(exc.body, bytes) else exc.body
            return f"CVAT rejected the request: {(body or exc.reason or '')[:300]}"
        return (
            f"CVAT answered {exc.status} {exc.reason}. Try again; if it repeats, call server_info."
        )
    return None


def tool_errors(fn: F) -> F:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            message = explain(exc)
            if message is None:
                raise
            raise ToolError(message) from exc

    return wrapper
