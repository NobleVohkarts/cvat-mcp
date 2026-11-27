"""MCP tasks (the io.modelcontextprotocol/tasks extension, protocol 2026-07-28): long calls answer
at once with a task and finish in the background; the client polls tasks/get for the result.

Every store access happens on the server's event loop, so the store needs no lock.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

import pydantic_core
from mcp.server.extension import Extension, MethodBinding
from mcp.shared.exceptions import MCPError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS
from mcp_types import CallToolResult, InputRequiredResult, RequestParams, TextContent
from mcp_types.version import MODERN_PROTOCOL_VERSIONS, is_version_at_least

from cvat_mcp.confirm import answered_yes, can_show_form, needs_confirmation
from cvat_mcp.services.auto_annotation import AutoAnnotation, cancel_run, watch_run

logger = logging.getLogger("cvat_mcp")

EXT = "io.modelcontextprotocol/tasks"
# tool -> whether a call with these arguments is long (auto_annotate with a frame waits seconds)
LONG_TOOLS: dict[str, Callable[[Mapping[str, Any]], bool]] = {
    "create_task": lambda _arguments: True,
    "import_annotations": lambda _arguments: True,
    "export_dataset": lambda _arguments: True,
    "create_report": lambda _arguments: True,
    "auto_annotate": lambda arguments: arguments.get("frame") is None,
}
TTL_MS = 3_600_000  # finished tasks are kept an hour
POLL_MS = 5_000
RUN_POLL_SECONDS = 1.0  # how often an auto_annotate task reads its run's progress
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
    run_id: str | None = None  # an auto_annotate run this task follows: cancel stops it
    cancel_requested: bool = False  # cancelled: before the run existed, or while it runs


def _iso(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat().replace("+00:00", "Z")


def _dump(result: CallToolResult) -> dict[str, Any]:
    return result.model_dump(by_alias=True, mode="json", exclude_none=True)


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
        if record.result is not None:  # set only with "completed"
            answer["result"] = record.result
        if record.error is not None:  # set only with "failed"
            answer["error"] = record.error
        return answer


class TaskParams(RequestParams):
    task_id: str  # taskId on the wire (mcp's models use camelCase aliases)


def becomes_task(params, ctx) -> bool:
    """A long call from a client that declared tasks on 2026-07-28, with no question still open:
    a call that asks first answers input_required as usual, and only its yes retry is a task."""
    name, arguments = params.name, params.arguments or {}
    long = LONG_TOOLS.get(name)
    if long is None or not long(arguments):
        return False
    if not is_version_at_least(ctx.protocol_version or "", MODERN_PROTOCOL_VERSIONS[0]):
        return False
    capabilities = ctx.session.client_capabilities
    if capabilities is None or EXT not in (capabilities.extensions or {}):
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
        running = asyncio.get_running_loop().create_task(self.finish(record, call_next(ctx)))
        self._running.add(running)
        running.add_done_callback(self._running.discard)
        return self.store.wire(record, created=True)

    async def finish(self, record: TaskRecord, call: Awaitable) -> None:
        """Runs the call to its end; any exception fails the task, or the client polls forever.
        A finished coroutine drops its frame, so a followed run doesn't keep the request alive."""
        try:
            result = await self._outcome(record, await call)
        except asyncio.CancelledError:  # the loop stops it (shutdown): don't leave it working
            self._fail(
                record, {"code": INTERNAL_ERROR, "message": "internal error: the call was stopped"}
            )
            raise
        except Exception as exc:  # pylint: disable=broad-exception-caught  # for the client
            if isinstance(exc, MCPError):
                error = exc.error.model_dump(mode="json", exclude_none=True)
            else:
                logger.exception("task %s (%s) failed", record.task_id, record.tool)
                error = {"code": INTERNAL_ERROR, "message": f"internal error: {type(exc).__name__}"}
            self._fail(record, error)
            return
        if result is not None:
            record.status_message = None  # the last progress line would contradict the result
            self.store.update(record, status="completed", result=_dump(result))

    def _fail(self, record: TaskRecord, error: dict) -> None:
        record.status_message = None
        self.store.update(record, status="failed", error=error)

    async def _outcome(self, record: TaskRecord, result) -> CallToolResult | None:
        """The task's final answer; None when it was cancelled."""
        if isinstance(result, InputRequiredResult):
            # shortcut: becomes_task saw every question answered yes; if the SDK still asks,
            # the call ends here and the model calls again
            text = "the confirmation was not answered; call again"
            return CallToolResult(is_error=True, content=[TextContent(type="text", text=text)])
        if record.tool == "auto_annotate" and not result.is_error:  # a whole-task run
            return await self._follow(
                record, AutoAnnotation.model_validate(result.structured_content)
            )
        return result

    async def _follow(self, record: TaskRecord, started: AutoAnnotation) -> CallToolResult | None:
        """The task ends when the run does."""
        record.run_id = started.run.run_id
        read = watch_run(record.run_id)  # bound to this run, even after a newer one starts
        if read is None:
            text = (
                f"{record.run_id} is no longer kept: get_auto_annotation_run shows the latest run"
            )
            return CallToolResult(is_error=True, content=[TextContent(type="text", text=text)])
        if record.cancel_requested:
            cancel_run(record.run_id)
        while (run := read()).state == "running":
            message = f"frames saved {run.frames_done} of {run.frames_total}"
            if record.cancel_requested:
                message += "; stopping after the current batch"
            self.store.update(record, status_message=message)
            await asyncio.sleep(RUN_POLL_SECONDS)
        if run.state == "cancelled":
            self.store.update(record, status="cancelled", status_message=run.error)
            return None
        seconds = (run.finished - run.started).total_seconds() if run.finished else started.seconds
        final = started.model_copy(
            update={"run": run, "note": None, "seconds": seconds, "skipped": dict(run.skipped)}
        )
        text = pydantic_core.to_json(final, indent=2).decode()  # as mcp writes a direct answer
        return CallToolResult(
            content=[TextContent(type="text", text=text)],
            structured_content=final.model_dump(mode="json"),
            is_error=run.state == "failed",
        )

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
        self.store.get(params.task_id)  # the spec lists it; no task here ever waits for input
        return {}

    async def _cancel(self, _ctx, params: TaskParams) -> dict[str, Any]:
        record = self.store.get(params.task_id)
        if record.status == "working":
            if record.tool == "auto_annotate" and record.run_id is None:  # still starting
                record.cancel_requested = True
                self.store.update(record, status_message="stopping once the run starts")
            elif record.run_id is None:
                self.store.update(record, status_message="CVAT can't stop this job; it will finish")
            elif cancel_run(record.run_id):
                record.cancel_requested = True  # the progress line keeps saying so
                self.store.update(record, status_message="stopping after the current batch")
        return {}
