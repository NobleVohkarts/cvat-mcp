"""MCP tasks (the io.modelcontextprotocol/tasks extension, protocol 2026-07-28): long calls answer
at once with a task and finish in the background; the client polls tasks/get for the result.

Every store access happens on the server's event loop, so the store needs no lock.
"""

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from mcp.server.extension import Extension, MethodBinding
from mcp.shared.exceptions import MCPError
from mcp.types import INVALID_PARAMS
from mcp_types import RequestParams
from mcp_types.version import MODERN_PROTOCOL_VERSIONS
from pydantic import Field

EXT = "io.modelcontextprotocol/tasks"
TTL_MS = 3_600_000  # finished tasks are kept an hour
POLL_MS = 5_000
Status = Literal["working", "completed", "failed", "cancelled"]


@dataclass
class TaskRecord:  # pylint: disable=too-many-instance-attributes
    task_id: str
    tool: str
    status: Status
    created: float
    updated: float
    status_message: str | None = None
    result: dict | None = None
    error: dict | None = None
    cancel: Callable[[], bool] | None = None  # True when the work will stop


def _iso(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat().replace("+00:00", "Z")


class TaskStore:
    def __init__(self, now: Callable[[], float] = time.time):
        self._now = now
        self._records: dict[str, TaskRecord] = {}

    def _sweep(self) -> None:
        cutoff = self._now() - TTL_MS / 1000
        for task_id, record in list(self._records.items()):
            if record.status != "working" and record.updated < cutoff:
                del self._records[task_id]

    def create(self, tool: str) -> TaskRecord:
        self._sweep()
        now = self._now()
        record = TaskRecord(uuid.uuid4().hex, tool, "working", now, now)
        self._records[record.task_id] = record
        return record

    def get(self, task_id: str) -> TaskRecord:
        self._sweep()
        record = self._records.get(task_id)
        if record is None:
            raise MCPError(code=INVALID_PARAMS, message=f"unknown task {task_id}")
        return record

    def update(
        self,
        record: TaskRecord,
        *,
        status: Status | None = None,
        status_message: str | None = None,
        result: dict | None = None,
        error: dict | None = None,
    ) -> None:
        if status is not None:
            record.status = status
        if status_message is not None:
            record.status_message = status_message
        if result is not None:
            record.result = result
        if error is not None:
            record.error = error
        record.updated = self._now()

    @staticmethod
    def wire(record: TaskRecord, *, created: bool = False) -> dict[str, Any]:
        """The create answer (`created`) or the tasks/get answer."""
        answer: dict[str, Any] = {
            "resultType": "task" if created else "complete",
            "taskId": record.task_id,
            "status": record.status,
            "createdAt": _iso(record.created),
            "lastUpdatedAt": _iso(record.updated),
            "ttlMs": TTL_MS,
            "pollIntervalMs": POLL_MS,
        }
        if record.status_message is not None:
            answer["statusMessage"] = record.status_message
        if record.status == "completed" and record.result is not None:
            answer["result"] = record.result
        if record.status == "failed" and record.error is not None:
            answer["error"] = record.error
        return answer


class TaskParams(RequestParams):
    task_id: str = Field(alias="taskId")


class Tasks(Extension):
    identifier = EXT

    def __init__(self, store: TaskStore | None = None):
        self.store = store or TaskStore()

    def methods(self):
        versions = frozenset(MODERN_PROTOCOL_VERSIONS)
        return [
            MethodBinding("tasks/get", TaskParams, self._get, versions),
            MethodBinding("tasks/update", TaskParams, self._update, versions),
            MethodBinding("tasks/cancel", TaskParams, self._cancel, versions),
        ]

    async def _get(self, _ctx, params: TaskParams) -> dict[str, Any]:
        return self.store.wire(self.store.get(params.task_id))

    async def _update(self, _ctx, params: TaskParams) -> dict[str, Any]:
        self.store.get(params.task_id)  # no task here ever waits for input
        return {}

    async def _cancel(self, _ctx, params: TaskParams) -> dict[str, Any]:
        record = self.store.get(params.task_id)
        if record.status == "working":
            if record.cancel is None:
                message = "CVAT can't stop this job; it will finish"
            elif record.cancel():
                message = "stopping after the current batch"
            else:
                message = None
            self.store.update(record, status_message=message)
        return {}
