"""Destructive calls ask the user to confirm through the client (MCP elicitation) when it can
show a form; clients that can't run them as before."""

from collections.abc import Callable, Mapping
from typing import Any

from mcp.server.elicitation import AcceptedElicitation, ElicitationResult
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.resolve import Elicit
from mcp_types import ClientCapabilities
from pydantic import BaseModel, Field


class Confirm(BaseModel):
    confirm: bool = Field(default=False, title="Yes, go ahead")  # left out: a decline


def _always(_arguments: Mapping[str, Any]) -> bool:
    return True


def _replace(arguments: Mapping[str, Any]) -> bool:
    return arguments.get("mode", "append") == "replace"


def _overwrite(arguments: Mapping[str, Any]) -> bool:
    return arguments.get("overwrite", False) is True


RISKY: dict[str, Callable[[Mapping[str, Any]], bool]] = {
    "delete_task": _always,
    "delete_gt_job": _always,
    "delete_cloud_storage": _always,
    "import_annotations": _replace,
    "auto_annotate": _replace,
    "export_dataset": _overwrite,
}

Answer = ElicitationResult[Confirm]  # as a parameter type, decline and cancel reach the tool body
YES: AcceptedElicitation[Confirm] = AcceptedElicitation(data=Confirm(confirm=True))  # the default


def needs_confirmation(tool: str, arguments: Mapping[str, Any]) -> bool:
    risky = RISKY.get(tool)
    return risky is not None and risky(arguments)


def can_show_form(capabilities: ClientCapabilities | None) -> bool:
    """The SDK's own rule (resolve._require_capability): bare `elicitation: {}` means form."""
    elicitation = capabilities.elicitation if capabilities is not None else None
    return elicitation is not None and (elicitation.form is not None or elicitation.url is None)


def ask(
    ctx: Context, tool: str, arguments: Mapping[str, Any], message: str
) -> Elicit[Confirm] | Confirm:
    """A resolver's answer: the question, or a yes without asking (the SDK wraps a plain value
    as accepted). An unknown tool name raises: a typo must not switch the form off."""
    if not RISKY[tool](arguments):
        return YES.data
    # A 2026-07-28 retry carries the answer: the SDK reads it only if the question is asked again,
    # so a retry must never skip it (e.g. a retry whose capabilities went missing).
    retry = ctx.input_responses is not None or ctx.request_state is not None
    if not retry and not can_show_form(ctx.client_capabilities):
        return YES.data
    return Elicit(message, Confirm)


def require_yes(answer: Answer, refusal: str) -> None:
    if not (isinstance(answer, AcceptedElicitation) and answer.data.confirm):
        raise ToolError(f"the user declined: {refusal}; don't call again unless the user asks")
