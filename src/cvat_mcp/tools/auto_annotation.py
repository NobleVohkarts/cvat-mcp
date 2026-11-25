from typing import Annotated

from mcp.server.mcpserver import Context
from pydantic import Field

from cvat_mcp.server import AUTO_ANNOTATION, get_client, open_client, tool
from cvat_mcp.services import auto_annotation
from cvat_mcp.services.auto_annotation import (
    AutoAnnotation,
    AutoAnnotationRuns,
    Function,
    Mode,
)

Frame = Annotated[
    int | None,
    Field(
        description="one task frame number: the call waits and returns the new objects; omit "
        "to annotate the whole task in the background"
    ),
]


# destructive: mode="replace"; open world: the first use of a model downloads its weights
@tool(write=True, destructive_hint=True, open_world_hint=True, enabled=AUTO_ANNOTATION)
def auto_annotate(
    ctx: Context,
    task_id: int,
    frame: Frame = None,
    function: Function = "detection",
    model: str | None = None,
    conf_threshold: float = 0.5,
    polygons: bool = True,
    mode: Mode = "append",
) -> AutoAnnotation:
    """Use to auto-annotate a task with a built-in torchvision model, only when the user asks:
    one frame, or the whole task. function: detection (boxes), instance_segmentation (polygons,
    or masks with polygons=false), keypoint_detection (skeletons; the task needs a skeleton
    label) or classification (tags). model: a torchvision model name for that function
    (default fasterrcnn_resnet50_fpn_v2, maskrcnn_resnet50_fpn_v2, keypointrcnn_resnet50_fpn,
    resnet50). The model's labels are matched to the task's by name; the others are skipped and
    counted. With frame the call waits (a few seconds) and returns the new objects: check them
    with render_frame. Without frame it starts a background run over the whole task (image
    tasks only, not video) and returns its run_id: follow it with get_auto_annotation_run; the
    run saves everything in one upload when it finishes. mode "append" adds objects (running
    twice duplicates them); "replace" swaps the shapes and tags on that frame (or, for a whole
    task, the task's, deleted after the upload) for the new ones, never tracks, only when the
    user asks. One run at a time. The first use of a model downloads its weights (~170 MB)."""
    opts = auto_annotation.check_options(function, model, conf_threshold, polygons, mode)
    client = get_client(ctx)
    if frame is None:
        return auto_annotation.start_run(client, open_client, task_id, opts)
    return auto_annotation.auto_annotate_frame(client, task_id, frame, opts)


@tool(write=True, read_only_hint=True, idempotent_hint=True, enabled=AUTO_ANNOTATION)
def get_auto_annotation_run(run_id: str | None = None) -> AutoAnnotationRuns:
    """Use to follow a background auto_annotate run: state (running, finished, failed), frames
    done out of the total, objects found per label (added) and skipped, and the error if it
    failed. Objects are saved only when the run finishes. Without run_id: every run since the
    server started, newest first. Runs live in memory: a server restart forgets them and stops
    a run in progress, which then saves nothing."""
    return auto_annotation.get_runs(run_id)
