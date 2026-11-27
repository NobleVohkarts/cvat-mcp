"""The tasks extension (io.modelcontextprotocol/tasks): store, methods, and the tools/call hook."""

import re
import time
from datetime import datetime, timezone
from typing import Annotated

import anyio
import pytest
from mcp import Client
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.resolve import Resolve
from mcp.shared.exceptions import MCPError
from mcp_types import ElicitResult
from tasks_client import TasksClient, send

from cvat_mcp import tasks as tasks_module
from cvat_mcp.confirm import YES, Answer, answered_yes, ask, require_yes
from cvat_mcp.services.auto_annotation import (
    AutoAnnotation,
    AutoAnnotationRun,
    AutoAnnotationRunStatus,
)
from cvat_mcp.tasks import POLL_MS, TTL_MS, Tasks, TaskStore

ISO_Z = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?Z$")


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


def test_create_then_get_working():
    store = TaskStore()
    record = store.create("create_report")
    created = store.wire(record, created=True)
    assert created["resultType"] == "task" and created["status"] == "working"
    assert re.fullmatch(r"[0-9a-f]{32}", created["taskId"])
    assert (created["ttlMs"], created["pollIntervalMs"]) == (TTL_MS, POLL_MS) == (3_600_000, 5_000)
    assert ISO_Z.match(created["createdAt"]) and ISO_Z.match(created["lastUpdatedAt"])
    got = store.wire(store.get(created["taskId"]))
    assert got["resultType"] == "complete" and got["status"] == "working"
    assert "result" not in got and "error" not in got


def test_completed_carries_result():
    store = TaskStore()
    record = store.create("create_report")
    result = {"content": [{"type": "text", "text": "ok"}], "isError": False}
    store.update(record, status="completed", result=result)
    got = store.wire(store.get(record.task_id))
    assert got["status"] == "completed" and got["result"] == result and "error" not in got


def test_failed_carries_error():
    store = TaskStore()
    record = store.create("create_report")
    error = {"code": -32603, "message": "internal error: RuntimeError"}
    store.update(record, status="failed", error=error)
    got = store.wire(store.get(record.task_id))
    assert got["status"] == "failed" and got["error"] == error and "result" not in got


def test_status_message_is_sent_when_set():
    store = TaskStore()
    record = store.create("auto_annotate")
    assert "statusMessage" not in store.wire(record)
    store.update(record, status_message="frames saved 10 of 20")
    assert store.wire(record)["statusMessage"] == "frames saved 10 of 20"


def test_unknown_task_is_invalid_params():
    with pytest.raises(MCPError) as caught:
        TaskStore().get("nope")
    assert caught.value.error.code == -32602 and caught.value.error.message == "unknown task nope"


def test_terminal_task_expires():
    clock = Clock()
    store = TaskStore(now=clock)
    record = store.create("create_report")
    store.update(record, status="completed", result={"content": []})
    clock.now += 3599
    assert store.get(record.task_id) is record
    clock.now += 2
    with pytest.raises(MCPError) as caught:
        store.get(record.task_id)
    assert caught.value.error.code == -32602


def test_working_task_never_expires():
    clock = Clock()
    store = TaskStore(now=clock)
    record = store.create("create_report")
    clock.now += 10_000
    assert store.get(record.task_id) is record


# The tools/call hook, on a standalone server whose fake tools carry the real long tools' names.
server = MCPServer("t", extensions=[Tasks()])
TOOL_SECONDS = 0.1


@server.tool()
def create_report(n: int) -> str:
    time.sleep(TOOL_SECONDS)
    return f"report {n}"


@server.tool()
def import_annotations(n: int) -> str:
    raise ToolError(f"bad file {n}")


@server.tool()
def list_tasks() -> str:
    return "tasks"


async def _confirm_overwrite(ctx: Context, overwrite: bool):
    return ask(ctx, "export_dataset", {"overwrite": overwrite}, "Overwrite?")


@server.tool()
def export_dataset(
    overwrite: bool = False, confirmation: Annotated[Answer, Resolve(_confirm_overwrite)] = YES
) -> str:
    require_yes(confirmation, "nothing was exported")
    return "exported"


@server.tool()
def auto_annotate(task_id: int, frame: int | None = None, delay: float = 0) -> AutoAnnotation:
    time.sleep(delay)  # start_run's checks and a first model download take this long
    started = {"task_id": task_id, "output": "boxes", "model": "m", "mode": "append", "seconds": 1}
    if frame is not None:
        return AutoAnnotation(**started, frame=frame)
    return AutoAnnotation(
        **started, note="follow it with get_auto_annotation_run", run=_run("running")
    )


def _run(state, done=0, error=None, run_id="run-1"):
    return AutoAnnotationRun(
        run_id=run_id,
        state=state,
        task_id=7,
        output="boxes",
        model="m",
        mode="append",
        frames_total=20,
        frames_done=done,
        started=datetime(2026, 11, 27, tzinfo=timezone.utc),
        error=error,
    ).model_dump(mode="json")


@pytest.fixture
def fake_run(monkeypatch):
    """get_run walks through `states`; cancel_run records its calls."""
    states, cancelled = [], []

    def get_run(run_id):
        state = states.pop(0) if len(states) > 1 else states[0]
        return AutoAnnotationRunStatus(run=AutoAnnotationRun.model_validate(state))

    def cancel_run(run_id):
        cancelled.append(run_id)
        states[:] = [
            _run("cancelled", 10, "stopped by a cancel; the first 10 of 20 frames are saved")
        ]
        return True

    monkeypatch.setattr(tasks_module, "get_run", get_run)
    monkeypatch.setattr(tasks_module, "cancel_run", cancel_run)
    monkeypatch.setattr(tasks_module, "RUN_POLL_SECONDS", 0.01)
    return states, cancelled


def tasks_client(mode="2026-07-28", tasks=None, **kw):
    return Client(server, mode=mode, extensions=[tasks] if tasks else [], **kw)


@pytest.mark.anyio
async def test_task_outlives_its_request():
    tasks = TasksClient()
    async with tasks_client(tasks=tasks) as client:
        started = time.monotonic()
        result = await client.call_tool("create_report", {"n": 2})
    assert not result.is_error and result.content[0].text == "report 2"
    assert tasks.seen[0][0] == "working"  # the first poll found it still running
    assert tasks.seen[-1][0] == "completed"
    assert time.monotonic() - started >= TOOL_SECONDS


@pytest.mark.anyio
async def test_tool_error_completes_with_is_error():
    tasks = TasksClient()
    async with tasks_client(tasks=tasks) as client:
        result = await client.call_tool("import_annotations", {"n": 3})
    assert result.is_error and result.content[0].text.endswith("bad file 3")
    assert tasks.seen[-1][0] == "completed"


@pytest.mark.anyio
async def test_unexpected_exception_fails():
    store = TaskStore()
    record = store.create("create_report")

    async def broken(_ctx):
        raise RuntimeError("boom")

    await Tasks(store).finish(record, broken(None))
    assert record.status == "failed"
    assert record.error == {"code": -32603, "message": "internal error: RuntimeError"}


@pytest.mark.anyio
async def test_short_tool_never_a_task():
    tasks = TasksClient()
    async with tasks_client(tasks=tasks) as client:
        assert (await client.call_tool("list_tasks", {})).content[0].text == "tasks"
    assert not tasks.seen


@pytest.mark.anyio
async def test_client_without_extension_gets_direct_result():
    async with tasks_client() as client:
        assert (await client.call_tool("create_report", {"n": 4})).content[0].text == "report 4"


@pytest.mark.anyio
async def test_legacy_protocol_never_gets_a_task():
    tasks = TasksClient()
    async with tasks_client(mode="legacy", tasks=tasks) as client:
        assert (await client.call_tool("create_report", {"n": 5})).content[0].text == "report 5"
    assert not tasks.seen


@pytest.mark.anyio
async def test_cancel_cvat_task_keeps_working():
    tasks = TasksClient()
    async with tasks_client(tasks=tasks) as client:
        async with anyio.create_task_group() as group:
            group.start_soon(client.call_tool, "create_report", {"n": 6})
            while not tasks.task_ids:
                await anyio.sleep(0.01)
            cancelled = await send(client.session, "tasks/cancel", tasks.task_ids[0])
            assert cancelled == {"resultType": "complete"}  # empty: mcp stamps the resultType
            state = await send(client.session, "tasks/get", tasks.task_ids[0])
            assert state["status"] == "working"
            assert state["statusMessage"] == "CVAT can't stop this job; it will finish"
    assert tasks.seen[-1][0] == "completed"


@pytest.mark.anyio
async def test_unknown_task_over_the_wire():
    async with tasks_client(tasks=TasksClient()) as client:
        with pytest.raises(MCPError) as caught:
            await send(client.session, "tasks/get", "nope")
    assert caught.value.error.code == -32602


@pytest.mark.anyio
async def test_methods_are_2026_only():
    async with tasks_client(mode="legacy") as client:
        with pytest.raises(MCPError) as caught:
            await send(client.session, "tasks/get", "nope")
    assert caught.value.error.code == -32601  # METHOD_NOT_FOUND


@pytest.mark.anyio
@pytest.mark.parametrize("action", ["accept", "decline"])
async def test_confirmation_before_task(action):
    tasks, asked = TasksClient(), []

    async def callback(_ctx, params):
        asked.append(params.message)
        content = {"confirm": True} if action == "accept" else None
        return ElicitResult(action=action, content=content)

    async with tasks_client(tasks=tasks, elicitation_callback=callback) as client:
        result = await client.call_tool("export_dataset", {"overwrite": True})
    assert asked == ["Overwrite?"]
    if action == "accept":
        assert result.content[0].text == "exported" and tasks.seen  # asked, then a task
    else:
        assert result.is_error and "the user declined" in result.content[0].text
        assert not tasks.seen  # a decline never starts a task


def test_answered_yes():
    yes = ElicitResult(action="accept", content={"confirm": True})
    assert answered_yes({"a": yes})
    assert not answered_yes({"a": ElicitResult(action="decline")})
    assert not answered_yes({"a": ElicitResult(action="accept", content={"confirm": False})})
    assert not answered_yes({"a": ElicitResult(action="accept", content={})})
    assert not answered_yes({})
    assert not answered_yes(None)


@pytest.mark.anyio
async def test_auto_annotate_task_follows_the_run(fake_run):
    states, _ = fake_run
    states[:] = [_run("running", 10), _run("running", 10), _run("finished", 20)]
    tasks = TasksClient(poll_seconds=0.01)
    async with tasks_client(tasks=tasks) as client:
        result = await client.call_tool("auto_annotate", {"task_id": 7})
    assert ("working", "frames saved 10 of 20") in tasks.seen
    assert tasks.seen[-1][0] == "completed"
    assert result.structured_content["run"]["state"] == "finished"
    assert result.structured_content["run"]["frames_done"] == 20
    assert result.structured_content["note"] is None


@pytest.mark.anyio
async def test_auto_annotate_cancel_ends_cancelled(fake_run):
    states, cancelled = fake_run
    states[:] = [_run("running", 10)]
    tasks = TasksClient(poll_seconds=0.01)
    async with tasks_client(tasks=tasks) as client:
        async with anyio.create_task_group() as group:
            outcome = {}

            async def call():
                outcome["result"] = await client.call_tool("auto_annotate", {"task_id": 7})

            group.start_soon(call)
            while not tasks.task_ids:
                await anyio.sleep(0.01)
            await send(client.session, "tasks/cancel", tasks.task_ids[0])
    assert cancelled == ["run-1"]
    assert tasks.seen[-1] == (
        "cancelled",
        "stopped by a cancel; the first 10 of 20 frames are saved",
    )
    assert outcome["result"].is_error


@pytest.mark.anyio
async def test_auto_annotate_with_frame_is_never_a_task():
    tasks = TasksClient()
    async with tasks_client(tasks=tasks) as client:
        result = await client.call_tool("auto_annotate", {"task_id": 7, "frame": 0})
    assert result.structured_content["frame"] == 0 and not tasks.seen


@pytest.mark.anyio
async def test_cancel_before_the_run_starts(fake_run):
    states, cancelled = fake_run
    states[:] = [_run("running", 0)]
    tasks = TasksClient(poll_seconds=0.01)
    async with tasks_client(tasks=tasks) as client:
        async with anyio.create_task_group() as group:
            group.start_soon(client.call_tool, "auto_annotate", {"task_id": 7, "delay": 0.3})
            while not tasks.task_ids:
                await anyio.sleep(0.01)
            await send(client.session, "tasks/cancel", tasks.task_ids[0])
            state = await send(client.session, "tasks/get", tasks.task_ids[0])
            assert state["statusMessage"] == "stopping once the run starts"
    assert cancelled == ["run-1"]  # applied as soon as the run existed
    assert tasks.seen[-1][0] == "cancelled"


@pytest.mark.anyio
async def test_follow_failure_fails_the_task(monkeypatch):
    def get_run(_run_id):
        raise RuntimeError("snapshot broke")

    monkeypatch.setattr(tasks_module, "get_run", get_run)
    async with tasks_client(tasks=TasksClient(poll_seconds=0.01)) as client:
        with pytest.raises(MCPError) as caught:
            await client.call_tool("auto_annotate", {"task_id": 7})
    assert caught.value.error.code == -32603
    assert caught.value.error.message == "internal error: RuntimeError"


@pytest.mark.anyio
async def test_follow_stops_at_another_run(fake_run):
    states, _ = fake_run
    states[:] = [_run("running", 10), _run("finished", 20, run_id="run-2")]
    async with tasks_client(tasks=TasksClient(poll_seconds=0.01)) as client:
        result = await client.call_tool("auto_annotate", {"task_id": 7})
    assert result.is_error and "run-1 is no longer kept" in result.content[0].text


@pytest.mark.anyio
async def test_failed_run_is_an_error(fake_run):
    states, _ = fake_run
    states[:] = [_run("failed", 10, "model crashed; the first 10 of 20 frames are saved")]
    async with tasks_client(tasks=TasksClient(poll_seconds=0.01)) as client:
        result = await client.call_tool("auto_annotate", {"task_id": 7})
    assert result.is_error and result.structured_content["run"]["state"] == "failed"
