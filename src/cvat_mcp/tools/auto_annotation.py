from typing import Annotated

from mcp.server.mcpserver import Context
from pydantic import Field

from cvat_mcp.server import AUTO_ANNOTATION, get_client, open_client, tool
from cvat_mcp.services import auto_annotation
from cvat_mcp.services.auto_annotation import (
    DETECTION_MODELS,
    OUTPUTS,
    AutoAnnotation,
    AutoAnnotationRuns,
    Mode,
    Output,
)

Frame = Annotated[
    int | None,
    Field(
        description="one task frame number: the call waits and returns the new objects; omit "
        "to annotate the whole task in the background"
    ),
]
OutputKind = Annotated[
    Output,
    Field(
        description="boxes (detection), polygons or masks (instance segmentation), skeletons "
        "(keypoints; the task needs a skeleton label), tags (classification)"
    ),
]
Model = Annotated[
    str | None,
    Field(
        description="a built-in torchvision model; default per output: "
        + ", ".join(f"{output} {spec.default}" for output, spec in OUTPUTS.items())
        + f". boxes: any of {', '.join(DETECTION_MODELS)}; polygons and masks: maskrcnn_*; "
        "skeletons: keypointrcnn_*; tags: any torchvision classification model (e.g. resnet50, "
        "convnext_tiny)"
    ),
]


# destructive: mode="replace"; open world: the first use of a model downloads its weights
@tool(write=True, destructive_hint=True, open_world_hint=True, enabled=AUTO_ANNOTATION)
def auto_annotate(
    ctx: Context,
    task_id: int,
    frame: Frame = None,
    output: OutputKind = "boxes",
    model: Model = None,
    conf_threshold: float = 0.5,
    mode: Mode = "append",
) -> AutoAnnotation:
    """Use to auto-annotate a task with a built-in torchvision model, only when the user asks:
    one frame, or the whole task. The model's labels are matched to the task's by name; the
    others are skipped and counted. With frame the call waits (a few seconds) and returns the new objects: check them
    with render_frame. Without frame it starts a background run over the whole task and
    returns its run_id: follow it with get_auto_annotation_run; the run saves every 10 frames,
    so a failed run keeps the frames saved so far. mode "append" adds objects (running twice
    duplicates them); "replace" swaps the shapes and tags on the run's frames for the new ones
    (each batch's old ones deleted after its new ones are saved), never tracks, only when the
    user asks. One run at a time. Tasks with overlapping jobs are refused. The first use of a
    model downloads its weights (~170 MB)."""
    opts = auto_annotation.check_options(output, model, conf_threshold, mode)
    client = get_client(ctx)
    if frame is None:
        return auto_annotation.start_run(client, open_client, task_id, opts)
    return auto_annotation.auto_annotate_frame(client, task_id, frame, opts)


@tool(write=True, read_only_hint=True, idempotent_hint=True, enabled=AUTO_ANNOTATION)
def get_auto_annotation_run(run_id: str | None = None) -> AutoAnnotationRuns:
    """Use to follow a background auto_annotate run: state (running, finished, failed), frames
    saved out of the total, objects saved per label (added) and skipped, and the error if it
    failed (with what was saved). Without run_id: every run since the server started, newest
    first. Runs live in memory: a server restart forgets them and stops a run in progress (the
    frames it saved stay)."""
    return auto_annotation.get_runs(run_id)
