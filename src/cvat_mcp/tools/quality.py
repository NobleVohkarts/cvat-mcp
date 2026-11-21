from typing import Annotated

from mcp.server.mcpserver import Context
from pydantic import Field

from cvat_mcp.server import get_client, settings, tool
from cvat_mcp.services import quality, quality_settings
from cvat_mcp.services.quality import Report
from cvat_mcp.services.quality_settings import QualitySettings


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
    starts with all disabled; the error says how to enable one). A score null with status
    "not_computed" means nothing was compared. ids are the jobs' objects with their kind (shape,
    tag or track; get_frame_annotations shows shapes and tags); gt_ids are inside the GT job,
    which render_frame does not draw. Creates a report record in CVAT (so the token must allow
    writes, even in read-only mode); never changes annotations."""
    return quality.create_report(get_client(ctx), settings.url, task_id, max_conflicts)


@tool(write=True)
def update_quality_settings(
    ctx: Context,
    task_id: int,
    enable: list[str] | None = None,
    disable: list[str] | None = None,
    use_project_settings: bool | None = None,
) -> QualitySettings:
    """Use to switch a task's quality requirements (e.g. 'Base rectangle') on or off for
    create_report, when the user asks or create_report says nothing is enabled and the user wants
    the report. Changes go where the task's settings live: its project's while
    use_project_settings is on (then every task sharing them changes too: shared_with_tasks),
    else its own. use_project_settings=false gives the task its own settings (CVAT's own ones
    start all off). Names must match exactly. Returns the settings as they are now."""
    return quality_settings.update_quality_settings(
        get_client(ctx), task_id, enable or [], disable or [], use_project_settings
    )
