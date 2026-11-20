from mcp.server.mcpserver import Context

from cvat_mcp.server import get_client, tool
from cvat_mcp.services import tasks
from cvat_mcp.services.tasks import TaskList, TaskOverview


@tool()
def list_tasks(
    ctx: Context, project_id: int | None = None, search: str | None = None, limit: int = 20
) -> TaskList:
    """Use to find tasks: returns up to `limit` tasks (newest first, max 100) with frame count,
    status and job progress, plus `total`, the number of all matching tasks. `search` is CVAT's
    free-text search: it matches task names but also project names, owners and assignees.
    Use get_task_overview for one task's labels, jobs and object counts."""
    return tasks.list_tasks(get_client(ctx), project_id, search, limit)


@tool()
def get_task_overview(ctx: Context, task_id: int) -> TaskOverview:
    """Use after list_tasks to understand one task: labels, jobs with their frame ranges, stage,
    state and assignee, and object counts per label. Needed before reviewing frames or creating
    issues: an issue goes to the job of type "annotation" whose frame range contains the frame
    (ground-truth and consensus jobs can overlap those ranges)."""
    return tasks.get_task_overview(get_client(ctx), task_id)
