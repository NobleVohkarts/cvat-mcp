from typing import Annotated

from mcp.server.mcpserver import Context
from pydantic import Field

from cvat_mcp.server import get_client, settings, tool
from cvat_mcp.services import quality
from cvat_mcp.services.quality import Report


# Changes no annotations, so it exists in both modes; it does add a report record in CVAT.
@tool(read_only_hint=False, destructive_hint=False, idempotent_hint=False)
def create_report(
    ctx: Context,
    task_id: int,
    max_conflicts: Annotated[int, Field(ge=0, le=500)] = 100,
) -> Report:
    """Use to compare a task's annotation jobs with its ground-truth job (CVAT quality control):
    computes a fresh report, waits for it (seconds for small tasks) and returns the counts, the
    scores and up to max_conflicts conflicts by frame. The GT job must hold the reference
    annotations in the same shape type as the jobs (a COCO file with segmentation imports as
    polygons), and the task's quality settings must enable a requirement for that type (CVAT
    starts with all disabled; the error says how to enable one). shape_ids are ids from
    get_frame_annotations; gt_shape_ids are inside the GT job, which render_frame does not draw.
    Creates a report record in CVAT; never changes annotations."""
    return quality.create_report(get_client(ctx), settings.url, task_id, max_conflicts)
