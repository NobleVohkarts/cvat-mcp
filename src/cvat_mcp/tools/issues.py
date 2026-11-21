from typing import Annotated

from mcp.server.mcpserver import Context
from pydantic import Field

from cvat_mcp.server import get_client, settings, tool
from cvat_mcp.services import issues
from cvat_mcp.services.issues import DeletedIssue, IssueList, IssueRef


@tool()
def list_issues(ctx: Context, task_id: int, frame: int | None = None) -> IssueList:
    """Use to see the issues already filed on a task, or on one frame if given, before filing
    or rewriting issues there: id, job, frame, position, message, resolved, author, number of
    comments, and own (filed by this server's CVAT user, so delete_issue can remove it)."""
    return issues.list_issues(get_client(ctx), task_id, frame)


@tool(write=True)
def create_issue(
    ctx: Context,
    job_id: int,
    frame: int,
    message: str,
    object_id: int | None = None,
    bbox: Annotated[list[float], Field(min_length=4, max_length=4)] | None = None,
) -> IssueRef:
    """Use to report one annotation problem to the annotator, only when the user asks to file
    issues (a request to review is not one): creates an issue on a frame of a job, shown in the
    CVAT UI at the given place. Pass object_id to mark an existing object's
    box, or bbox [x1, y1, x2, y2] (original pixels, e.g. from render_frame's legend) to mark a
    region, e.g. where an object is missing. The job must be the annotation job whose frame
    range contains the frame (see get_task_overview). One issue per problem; the message should
    say what is wrong and what to do."""
    return issues.create_issue(
        get_client(ctx), settings.url, job_id, frame, message, object_id, bbox
    )


@tool(write=True, destructive_hint=True, idempotent_hint=True)
def delete_issue(ctx: Context, issue_id: int) -> DeletedIssue:
    """Use to remove an issue that is wrong or replaced, e.g. when the user asks to rewrite the
    issues on a frame: delete the old ones (ids from list_issues), then create_issue the new
    ones. Deletes the issue with all its comments. Only issues with own=true (filed by this
    server's CVAT user) can be deleted; others are refused and must be handled in the CVAT UI."""
    return issues.delete_issue(get_client(ctx), issue_id)
