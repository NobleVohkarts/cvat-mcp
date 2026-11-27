"""A test client for MCP tasks: claims `resultType: "task"`, polls tasks/get until the task ends."""

from typing import Literal

import anyio
from mcp.client.extension import ClientExtension, ResultClaim
from mcp.shared.exceptions import MCPError
from mcp.types import ErrorData
from mcp_types import CallToolResult, Request, Result
from pydantic import ConfigDict

EXT = "io.modelcontextprotocol/tasks"


class TaskCreated(Result):
    model_config = ConfigDict(extra="allow")
    result_type: Literal["task"] = "task"


class TaskAnswer(Result):
    model_config = ConfigDict(extra="allow")


class TaskRequest(Request):
    method: Literal["tasks/get", "tasks/cancel", "tasks/update"]
    params: dict


async def send(session, method: str, task_id: str) -> dict:
    answer = await session.send_request(
        TaskRequest(method=method, params={"taskId": task_id}), TaskAnswer
    )
    return answer.model_extra or {}


class TasksClient(ClientExtension):
    """`seen` records every (status, statusMessage) polled; `task_ids` every task created."""

    identifier = EXT

    def __init__(self, poll_seconds: float = 0.05):
        self.poll_seconds = poll_seconds
        self.seen: list[tuple[str, str | None]] = []
        self.task_ids: list[str] = []

    def claims(self):
        return [ResultClaim(result_type="task", model=TaskCreated, resolve=self._resolve)]

    async def _resolve(self, created: TaskCreated, ctx) -> CallToolResult:
        task_id = created.model_extra["taskId"]
        self.task_ids.append(task_id)
        while True:
            state = await send(ctx.session, "tasks/get", task_id)
            self.seen.append((state["status"], state.get("statusMessage")))
            if state["status"] == "completed":
                return CallToolResult.model_validate(state["result"])
            if state["status"] == "failed":
                raise MCPError.from_error_data(ErrorData.model_validate(state["error"]))
            if state["status"] == "cancelled":
                return CallToolResult.model_validate(state["result"])
            await anyio.sleep(self.poll_seconds)
