"""The tasks extension (io.modelcontextprotocol/tasks): store, methods, and the tools/call hook."""

import re

import pytest
from mcp.shared.exceptions import MCPError

from cvat_mcp.tasks import POLL_MS, TTL_MS, TaskStore

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
