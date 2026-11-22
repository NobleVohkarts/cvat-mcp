from typing import Annotated

from mcp.server.mcpserver import Context
from pydantic import Field

from cvat_mcp.server import AUTO_ANNOTATION, get_client, open_client, settings, tool
from cvat_mcp.services import auto_annotation
from cvat_mcp.services.auto_annotation import (
    AutoAnnotation,
    AutoAnnotationRuns,
    Function,
    Mode,
)

Frames = Annotated[
    list[int] | None,
    Field(
        description="task frame numbers (1 to 20): the call waits and returns the new objects; "
        "omit to annotate all the task's (or job's) frames in the background"
    ),
]


# destructive: mode="replace"; open world: the first use of a model downloads its weights
@tool(write=True, destructive_hint=True, open_world_hint=True, enabled=AUTO_ANNOTATION)
def auto_annotate(
    ctx: Context,
    task_id: int | None = None,
    job_id: int | None = None,
    frames: Frames = None,
    function: Function = "detection",
    model: str | None = None,
    conf_threshold: float = 0.5,
    polygons: bool = True,
    mode: Mode = "append",
) -> AutoAnnotation:
    """Use to auto-annotate with a built-in torchvision model, only when the user asks: a task
    (task_id) or one of its jobs (job_id), exactly one. function: detection (boxes),
    instance_segmentation (polygons, or masks with polygons=false), keypoint_detection
    (skeletons; the task needs a skeleton label) or classification (tags). model: a torchvision
    model name for that function (default fasterrcnn_resnet50_fpn_v2, maskrcnn_resnet50_fpn_v2,
    keypointrcnn_resnet50_fpn, resnet50). The model's labels are matched to the task's by name;
    the others are skipped and counted. With frames the call waits (a few seconds per frame) and
    returns the new objects: check them with render_frame. Without frames it starts a background
    run and returns its run_id: follow it with get_auto_annotation_run. Results are saved in
    chunks as they finish. mode "append" adds objects (running twice duplicates them);
    "replace" first deletes the shapes and tags on the run's frames (never tracks), only when the
    user asks. One run at a time. The first use of a model downloads its weights (~170 MB)."""
    opts = auto_annotation.check_options(function, model, conf_threshold, polygons, mode)
    client, size = get_client(ctx), settings.mcp_aa_chunk_size
    if frames is None:
        return auto_annotation.start_run(client, open_client, task_id, job_id, opts, size)
    return auto_annotation.auto_annotate_frames(client, task_id, job_id, frames, opts, size)


@tool(write=True, read_only_hint=True, idempotent_hint=True, enabled=AUTO_ANNOTATION)
def get_auto_annotation_run(run_id: str | None = None) -> AutoAnnotationRuns:
    """Use to follow a background auto_annotate run: state (running, finished, failed), frames
    done out of the total, objects added and skipped per label, and the error if it failed.
    Without run_id: every run since the server started, newest first. Runs live in memory: a
    server restart forgets them and stops a run in progress (the frames it saved stay)."""
    return auto_annotation.get_runs(run_id)
