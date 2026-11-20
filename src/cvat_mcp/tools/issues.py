from typing import Annotated

from mcp.server.mcpserver import Context
from pydantic import Field

from cvat_mcp.server import get_client, settings, tool
from cvat_mcp.services import issues
from cvat_mcp.services.issues import IssueRef


@tool(write=True)
def create_issue(
    ctx: Context,
    job_id: int,
    frame: int,
    message: str,
    object_id: int | None = None,
    bbox: Annotated[list[float], Field(min_length=4, max_length=4)] | None = None,
) -> IssueRef:
    """Use to report one annotation problem to the annotator: creates an issue on a frame of a
    job, shown in the CVAT UI at the given place. Pass object_id to mark an existing object's
    box, or bbox [x1, y1, x2, y2] (original pixels, e.g. from render_frame's legend) to mark a
    region, e.g. where an object is missing. The job must be the annotation job whose frame
    range contains the frame (see get_task_overview). One issue per problem; the message should
    say what is wrong and what to do."""
    return issues.create_issue(
        get_client(ctx), settings.url, job_id, frame, message, object_id, bbox
    )
