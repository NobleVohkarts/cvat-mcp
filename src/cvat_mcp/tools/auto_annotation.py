from typing import Annotated

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.resolve import Resolve
from pydantic import Field

from cvat_mcp.confirm import YES, Answer, ask, require_yes
from cvat_mcp.errors import InvalidArgument
from cvat_mcp.server import AUTO_ANNOTATION, get_client, open_client, tool
from cvat_mcp.services import auto_annotation
from cvat_mcp.services.auto_annotation import (
    OUTPUTS,
    AutoAnnotation,
    AutoAnnotationRunStatus,
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
        # Patterns, not torchvision's list: importing it here would cost every server start
        # seconds and hundreds of MB. A wrong name gets the installed models' list back.
        + ". Allowed: "
        + "; ".join(
            f"{output}: "
            + (
                "any torchvision classification model (e.g. resnet50, convnext_tiny)"
                if spec.models is None
                else (
                    "any torchvision detection model (e.g. fasterrcnn_resnet50_fpn_v2, "
                    "retinanet_resnet50_fpn_v2, ssdlite320_mobilenet_v3_large)"
                    if spec.models == "*"
                    else spec.models
                )
            )
            for output, spec in OUTPUTS.items()
        )
    ),
]


async def _confirm_replace(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    ctx: Context,
    task_id: int,
    frame: int | None,
    output: Output,
    model: str | None,
    conf_threshold: float,
    mode: Mode,
):
    try:
        auto_annotation.check_options(output, model, conf_threshold, mode)
    except InvalidArgument:
        return YES.data  # nothing to confirm: the body refuses the options before any change
    scope = (
        f"frame {frame} of task {task_id}" if frame is not None else f"all frames of task {task_id}"
    )
    return ask(
        ctx,
        "auto_annotate",
        {"mode": mode},
        f"Replace the shapes and tags on {scope} with the model's?",
    )


# destructive: mode="replace"; open world: the first use of a model downloads its weights
@tool(write=True, destructive_hint=True, open_world_hint=True, enabled=AUTO_ANNOTATION)
def auto_annotate(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    ctx: Context,
    task_id: int,
    frame: Frame = None,
    output: OutputKind = "boxes",
    model: Model = None,
    conf_threshold: float = 0.5,
    mode: Mode = "append",
    confirmation: Annotated[Answer, Resolve(_confirm_replace)] = YES,
) -> AutoAnnotation:
    """Use to auto-annotate a task with a built-in torchvision model, only when the user asks:
    one frame, or the whole task. The model's labels are matched to the task's by name; the
    others are skipped and counted. With frame the call waits (a few seconds; ~15 s more when
    it loads a model) and returns the new objects: check them with render_frame. Without frame
    it starts a background run over the whole task and returns its run_id: follow it with
    get_auto_annotation_run; the run saves every 10 frames, so a failed run keeps the frames
    saved so far. mode "append" adds objects (running twice
    duplicates them); "replace" swaps the shapes and tags on the run's frames for the new ones
    (each batch's old ones deleted after its new ones are saved), never tracks, only when the
    user asks. One run at a time. Tasks with overlapping jobs are refused. The first use of a
    model downloads its weights (~170 MB)."""
    require_yes(confirmation, f"nothing was annotated on task {task_id}")
    opts = auto_annotation.check_options(output, model, conf_threshold, mode)
    client = get_client(ctx)
    if frame is None:
        return auto_annotation.start_run(client, open_client, task_id, opts)
    return auto_annotation.auto_annotate_frame(client, task_id, frame, opts)


@tool(write=True, read_only_hint=True, idempotent_hint=True, enabled=AUTO_ANNOTATION)
def get_auto_annotation_run(run_id: str | None = None) -> AutoAnnotationRunStatus:
    """Use to follow a background auto_annotate run: state (running, finished, failed, or
    cancelled: stopped by a task cancel after its current batch, the frames saved stay), frames
    saved out of the total, objects saved per label (added) and skipped, and the error if it
    failed (with what was saved). Only the latest run is kept, in memory: an older run_id gets
    the latest run, and a server restart forgets it and stops it if it is running (the frames
    it saved stay)."""
    return auto_annotation.get_run(run_id)
