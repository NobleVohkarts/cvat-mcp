"""MCP tasks (the io.modelcontextprotocol/tasks extension, protocol 2026-07-28): long calls answer
at once with a task and finish in the background; the client polls tasks/get for the result.

Every store access happens on the server's event loop, so the store needs no lock.
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from mcp.server.extension import Extension, MethodBinding
from mcp.shared.exceptions import MCPError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS
from mcp_types import CallToolResult, InputRequiredResult, RequestParams, TextContent
from mcp_types.version import MODERN_PROTOCOL_VERSIONS, is_version_at_least
from pydantic import Field

from cvat_mcp.confirm import answered_yes, can_show_form, needs_confirmation
from cvat_mcp.services.auto_annotation import cancel_run, get_run

logger = logging.getLogger("cvat_mcp")
RUN_POLL_SECONDS = 1.0

EXT = "io.modelcontextprotocol/tasks"
LONG_TOOLS = frozenset(
    {"create_task", "import_annotations", "export_dataset", "create_report", "auto_annotate"}
)
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


def becomes_task(params, ctx) -> bool:
    """A long call from a client that declared tasks on 2026-07-28, with no question still open:
    a call that asks first answers input_required as usual, and only its yes retry is a task."""
    name, arguments = params.name, params.arguments or {}
    if name not in LONG_TOOLS or (name == "auto_annotate" and arguments.get("frame") is not None):
        return False
    if not is_version_at_least(ctx.protocol_version or "", MODERN_PROTOCOL_VERSIONS[0]):
        return False
    capabilities = ctx.session.client_capabilities
    if EXT not in ((capabilities.extensions or {}) if capabilities is not None else {}):
        return False
    if needs_confirmation(name, arguments) and can_show_form(capabilities):
        return answered_yes(params.input_responses)
    return True


class Tasks(Extension):
    identifier = EXT

    def __init__(self, store: TaskStore | None = None):
        self.store = store or TaskStore()
        self._running: set[asyncio.Task] = set()  # strong references until each call ends

    async def intercept_tool_call(self, params, ctx, call_next):
        if not becomes_task(params, ctx):
            return await call_next(ctx)
        record = self.store.create(params.name)
        # The call outlives this request (checked over stdio and streamable HTTP).
        running = asyncio.get_running_loop().create_task(self.finish(record, call_next, ctx))
        self._running.add(running)
        running.add_done_callback(self._running.discard)
        return self.store.wire(record, created=True)

    async def finish(self, record: TaskRecord, call_next, ctx) -> None:
        try:
            result = await call_next(ctx)
        except MCPError as exc:
            self.store.update(
                record,
                status="failed",
                error={"code": exc.error.code, "message": exc.error.message},
            )
            return
        except (
            Exception
        ) as exc:  # pylint: disable=broad-exception-caught  # recorded for the client
            logger.exception("task %s (%s) failed", record.task_id, record.tool)
            error = {"code": INTERNAL_ERROR, "message": f"internal error: {type(exc).__name__}"}
            self.store.update(record, status="failed", error=error)
            return
        if isinstance(result, InputRequiredResult):
            # shortcut: becomes_task saw every question answered yes; if the SDK still asks,
            # the call ends here and the model calls again
            text = "the confirmation was not answered; call again"
            result = CallToolResult(is_error=True, content=[TextContent(type="text", text=text)])
        if isinstance(result, CallToolResult):
            run = (result.structured_content or {}).get("run") if not result.is_error else None
            if record.tool == "auto_annotate" and run:
                await self._follow(record, result.structured_content, run["run_id"])
                return
            result = result.model_dump(by_alias=True, mode="json", exclude_none=True)
        self.store.update(record, status="completed", result=result)

    async def _follow(self, record: TaskRecord, started: dict, run_id: str) -> None:
        """A whole-task auto_annotate returned its run at once: the task ends when the run does."""
        record.cancel = lambda: cancel_run(run_id)
        while (run := get_run(run_id).run) is not None and run.state == "running":
            message = f"frames saved {run.frames_done} of {run.frames_total}"
            self.store.update(record, status_message=message)
            await asyncio.sleep(RUN_POLL_SECONDS)
        if run is None:  # shortcut: only a restart forgets a run, and that forgets this task too
            run_state = started["run"]
        else:
            run_state = run.model_dump(mode="json")
        if run_state["state"] == "cancelled":
            self.store.update(record, status="cancelled", status_message=run_state["error"])
            return
        final = started | {"run": run_state, "note": None}
        result = CallToolResult(
            content=[TextContent(type="text", text=json.dumps(final))], structured_content=final
        )
        dumped = result.model_dump(by_alias=True, mode="json", exclude_none=True)
        self.store.update(record, status="completed", result=dumped)

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
