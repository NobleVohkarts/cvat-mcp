"""Confirmation through elicitation, on a standalone server (no CVAT, no cvat_mcp.server)."""

from types import SimpleNamespace
from typing import Annotated

import pytest
from mcp import Client
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.resolve import Elicit, Resolve
from mcp_types import (
    ClientCapabilities,
    ElicitationCapability,
    ElicitResult,
    FormElicitationCapability,
    UrlElicitationCapability,
)

from cvat_mcp.confirm import YES, Answer, ask, can_show_form, needs_confirmation, require_yes

pytestmark = pytest.mark.anyio
MODES = ["2026-07-28", "legacy"]
DECLINED = ": the user declined: 7 was not replaced; don't call again unless the user asks"


def _confirm_drop(ctx: Context, x: int, mode: str = "append"):
    return ask(ctx, "import_annotations", {"mode": mode}, f"Replace {x}?")


server = MCPServer("t")


@server.tool()
def drop(
    x: int, mode: str = "append", answer: Annotated[Answer, Resolve(_confirm_drop)] = YES
) -> str:
    require_yes(answer, f"{x} was not replaced")
    return "done"


def answering(action, content=None):
    asked = []

    async def callback(_ctx, params):
        asked.append(params.message)
        return ElicitResult(action=action, content=content)

    return callback, asked


async def call(mode, arguments, callback=None):
    async with Client(server, mode=mode, elicitation_callback=callback) as client:
        result = await client.call_tool("drop", arguments)
    return result.is_error, result.content[0].text


@pytest.mark.parametrize("mode", MODES)
async def test_accept_runs(mode):
    callback, asked = answering("accept", {"confirm": True})
    assert await call(mode, {"x": 7, "mode": "replace"}, callback) == (False, "done")
    assert asked == ["Replace 7?"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("action", ["decline", "cancel"])
async def test_decline_and_cancel_refuse(mode, action):
    callback, asked = answering(action)
    is_error, text = await call(mode, {"x": 7, "mode": "replace"}, callback)
    assert is_error and text.endswith(DECLINED)  # after mcp's prefix
    assert asked == ["Replace 7?"]


@pytest.mark.parametrize("mode", MODES)
async def test_accept_unchecked_is_a_decline(mode):
    callback, _ = answering("accept", {"confirm": False})
    is_error, text = await call(mode, {"x": 7, "mode": "replace"}, callback)
    assert is_error and text.endswith(DECLINED)


@pytest.mark.parametrize("mode", MODES)
async def test_accept_without_the_field_is_a_decline(mode):  # e.g. a box left unticked
    callback, _ = answering("accept", {})
    is_error, text = await call(mode, {"x": 7, "mode": "replace"}, callback)
    assert is_error and text.endswith(DECLINED)


def test_unknown_tool_name_is_an_error():  # a typo must not switch the form off silently
    with pytest.raises(KeyError):
        ask(None, "delete_tsak", {}, "Delete?")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("arguments", [{"x": 7, "mode": "append"}, {"x": 7}])
async def test_not_risky_asks_nothing(mode, arguments):
    callback, asked = answering("decline")
    assert await call(mode, arguments, callback) == (False, "done")
    assert not asked


@pytest.mark.parametrize("mode", MODES)
async def test_no_form_capability_runs_unasked(mode):
    assert await call(mode, {"x": 7, "mode": "replace"}) == (False, "done")


async def test_resolver_parameter_is_hidden():
    (tool,) = await server.list_tools()
    assert set(tool.input_schema["properties"]) == {"x", "mode"}


def test_needs_confirmation():
    assert needs_confirmation("delete_task", {"task_id": 1})
    assert needs_confirmation("delete_gt_job", {"job_id": 1})
    assert needs_confirmation("delete_cloud_storage", {"cloud_storage_id": 1})
    assert needs_confirmation("import_annotations", {"mode": "replace"})
    assert not needs_confirmation("import_annotations", {})
    assert not needs_confirmation("import_annotations", {"mode": "append"})
    assert needs_confirmation("auto_annotate", {"task_id": 1, "mode": "replace"})
    assert not needs_confirmation("auto_annotate", {"task_id": 1})
    assert needs_confirmation("export_dataset", {"overwrite": True})
    assert not needs_confirmation("export_dataset", {"overwrite": False})
    assert not needs_confirmation("export_dataset", {})
    assert not needs_confirmation("delete_issue", {"issue_id": 1})
    assert not needs_confirmation("delete_annotations", {"task_id": 1})


def test_can_show_form():
    form, url = FormElicitationCapability(), UrlElicitationCapability()

    def caps(elicitation):
        return ClientCapabilities(elicitation=elicitation)

    assert not can_show_form(None)
    assert not can_show_form(ClientCapabilities())
    assert can_show_form(caps(ElicitationCapability()))  # bare {}: form, as on older revisions
    assert can_show_form(caps(ElicitationCapability(form=form)))
    assert can_show_form(caps(ElicitationCapability(form=form, url=url)))
    assert not can_show_form(caps(ElicitationCapability(url=url)))


@pytest.mark.parametrize(
    "retry",
    [
        {"input_responses": {"q": ElicitResult(action="decline")}, "request_state": "sealed"},
        {"input_responses": None, "request_state": "sealed"},
    ],
)
def test_a_retry_always_asks_again(retry):
    """On a 2026-07-28 retry the recorded answer is consulted only if the resolver asks again:
    a retry whose capabilities went missing must not turn a decline into a yes."""
    ctx = SimpleNamespace(client_capabilities=None, **retry)
    assert isinstance(ask(ctx, "delete_task", {"task_id": 7}, "Delete?"), Elicit)


def test_first_round_without_a_form_asks_nothing():
    ctx = SimpleNamespace(client_capabilities=None, input_responses=None, request_state=None)
    assert not isinstance(ask(ctx, "delete_task", {"task_id": 7}, "Delete?"), Elicit)
