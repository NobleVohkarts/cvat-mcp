import importlib
import itertools
import logging
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

import PIL.Image
from cvat_sdk import Client
from cvat_sdk.api_client import models
from cvat_sdk.auto_annotation import DetectionFunctionContext
from pydantic import BaseModel

from cvat_mcp.errors import InvalidArgument, explain
from cvat_mcp.services.annotation_edits import (
    REQUEST,
    check_editable,
    create_objects,
    patch_annotations,
)
from cvat_mcp.services.annotations import Scope, shape_bbox, target_scope

Function = Literal["detection", "instance_segmentation", "keypoint_detection", "classification"]
Mode = Literal["append", "replace"]

# REVIEW: why do we need these predefine models? have we documented it? how would mcp server caller know about this feature?
DEFAULT_MODELS: dict[str, str] = {
    "detection": "fasterrcnn_resnet50_fpn_v2",
    "instance_segmentation": "maskrcnn_resnet50_fpn_v2",
    "keypoint_detection": "keypointrcnn_resnet50_fpn",
    "classification": "resnet50",
}
# Detection models that also fit the segmentation / keypoint functions.
# REVIEW: why do we split a model map in two dict? Even if it's a part of model name it's better to use one structure with some flag or wildcard in the name
_MODEL_PART = {"instance_segmentation": "maskrcnn", "keypoint_detection": "keypointrcnn"}

logger = logging.getLogger(__name__)

# Frames per save: one request each, far below CVAT's 100 MB body and 60 s answer limits.
FRAMES_PER_SAVE = 10


@dataclass(frozen=True)
class RunOptions:
    function: Function
    model: str
    conf_threshold: float
    polygons: bool
    mode: Mode


# slots: the generated slot descriptors implement the interface's abstract properties
@dataclass(frozen=True, slots=True)
class Context(DetectionFunctionContext):
    frame_name: str
    conf_threshold: float | None
    conv_mask_to_poly: bool


class AutoObject(BaseModel):
    id: int | None  # None: CVAT's answer could not be matched; see get_frame_annotations
    kind: Literal["shape", "tag"]
    frame: int
    label: str
    type: str
    bbox: list[float] | None = None


class AutoAnnotationRun(BaseModel):
    run_id: str
    state: Literal["running", "finished", "failed"]
    task_id: int
    function: str
    model: str
    mode: str
    frames_total: int
    frames_done: int = 0
    added: dict[str, int] = {}
    skipped: dict[str, int] = {}
    started: datetime
    finished: datetime | None = None
    error: str | None = None


class AutoAnnotation(BaseModel):
    task_id: int
    function: str
    model: str
    mode: str
    frame: int | None = None
    objects: list[AutoObject] = []
    skipped: dict[str, int] = {}  # the function's label -> detections dropped (no such task label)
    seconds: float
    note: str | None = None
    run: AutoAnnotationRun | None = None  # set for a background run (no frame given)


@dataclass
class Progress:
    keep_objects: bool = True  # False for background runs: nobody reads them, and they'd pile up
    added: Counter = field(default_factory=Counter)  # task label -> saved objects
    skipped: Counter = field(default_factory=Counter)  # function label -> dropped (no task label)
    objects: list[AutoObject] = field(default_factory=list)
    frames_done: int = 0  # frames saved, in order


def check_options(
    function: Function, model: str | None, conf_threshold: float, polygons: bool, mode: Mode
) -> RunOptions:
    if not 0 <= conf_threshold <= 1:
        raise InvalidArgument("conf_threshold must be between 0 and 1")
    return RunOptions(function, model or DEFAULT_MODELS[function], conf_threshold, polygons, mode)


def _model_choices(function: str) -> list[str]:
    import torchvision.models  # pylint: disable=import-outside-toplevel  # optional extra

    if function == "classification":
        return torchvision.models.list_models(module=torchvision.models)
    part = _MODEL_PART.get(function, "")
    return [
        m for m in torchvision.models.list_models(module=torchvision.models.detection) if part in m
    ]


# REVIEW: it seems like a cache for me, why don't it instead?
_loaded: tuple[tuple[str, str], object] | None = None


def load_function(opts: RunOptions):
    """The built-in function with that model; the last one used stays loaded."""
    global _loaded  # pylint: disable=global-statement
    key = (opts.function, opts.model)
    if _loaded is not None and _loaded[0] == key:
        return _loaded[1]
    choices = _model_choices(opts.function)
    if opts.model not in choices:
        raise InvalidArgument(
            f"unknown {opts.function} model {opts.model!r}; choose one of: {', '.join(choices)}"
        )
    _loaded = None  # free the old model before loading the new one
    module = importlib.import_module(f"cvat_sdk.auto_annotation.functions.torchvision_{key[0]}")
    fn = module.create(opts.model)
    _loaded = (key, fn)
    return fn


def _old_objects(scope: Scope, frames: list[int]) -> dict[int, list]:
    """The scope's shapes and tags on those frames, as delete requests by frame; tracks stay."""
    wanted = set(frames)
    annotations = scope.source.get_annotations()
    by_frame = defaultdict(list)
    for kind, objs in (("shape", annotations.shapes), ("tag", annotations.tags)):
        for obj in objs:
            if obj.frame in wanted:
                by_frame[obj.frame].append((kind, REQUEST[kind](**obj.to_dict())))
    return by_frame


def _delete_stale(client: Client, scope: Scope, stale: list, saved: str) -> None:
    """Deletes the old objects once the new ones are saved; a failure says both are there now."""
    try:
        patch_annotations(client, "delete", scope, stale)
    except Exception as exc:
        # Only replace deletes, and a repeated replace removes the old and the new ones alike.
        raise InvalidArgument(
            f"the new objects were saved ({saved}), but deleting the {len(stale)} old shapes and "
            f"tags failed: {explain(exc) or exc}; both are there now, and repeating the call "
            "replaces them again"
        ) from exc


def _misfit(detail: str) -> InvalidArgument:
    return InvalidArgument(
        f"the model's output does not fit this task's labels: {detail}. The task's label types "
        "must fit what the function outputs (e.g. a skeleton label for keypoint_detection, or "
        "type any)"
    )


def label_map(fn, labels, polygons: bool) -> dict[int, tuple[int, dict[int, int]] | None]:
    """Function label id -> (task label id, function point id -> task point id), matched by
    name; None where the task has no such label. Refuses outputs the task's labels can't hold."""
    by_name = {label.name: label for label in labels}
    mapping: dict[int, tuple[int, dict[int, int]] | None] = {}
    for spec in fn.spec.labels:
        label = by_name.get(spec.name)
        if label is None:
            mapping[spec.id] = None
            continue
        output = getattr(spec, "type", "any")
        if output == "mask" and polygons:
            output = "polygon"
        if (output == "skeleton") != (label.type == "skeleton") or (
            output != "any" and label.type not in ("any", output)
        ):
            raise _misfit(f"label {spec.name!r} is {output!r} in the model, {label.type!r} here")
        points = {p.name: p.id for p in getattr(spec, "sublabels", None) or []}
        task_points = {p.name: p.id for p in getattr(label, "sublabels", None) or []}
        if missing := sorted(set(task_points) - set(points)):
            raise _misfit(
                f"skeleton {spec.name!r} has points the model does not output: {', '.join(missing)}"
            )
        mapping[spec.id] = (label.id, {points[name]: i for name, i in task_points.items()})
    return mapping


def remap(output, mapping, frame: int) -> list[tuple[str, object]]:
    """The function's output as (kind, request) for the task's labels on that frame; objects of
    labels the task lacks are dropped, and so are skeleton points it lacks."""
    sent = []
    for obj in output:
        if obj.label_id not in mapping:
            raise _misfit(f"the model output an unknown label id {obj.label_id}")
        if (target := mapping[obj.label_id]) is None:
            continue
        label_id, points = target
        obj.label_id, obj.frame, obj.source = label_id, frame, "auto"
        obj.attributes = []  # built-in functions output none; their ids would not be the task's
        elements = getattr(obj, "elements", None) or []
        if elements:
            obj.elements = [e for e in elements if e.label_id in points]
            for e in obj.elements:
                e.label_id, e.frame, e.source = points[e.label_id], frame, "auto"
        sent.append(("tag" if isinstance(obj, models.LabeledImageRequest) else "shape", obj))
    return sent


def _objects(sent, ids, names: dict[int, str]) -> list[AutoObject]:
    return [
        AutoObject(
            id=ids[i],
            kind=kind,
            frame=r.frame,
            label=names[r.label_id],
            type=str(r.type) if kind == "shape" else "tag",
            bbox=shape_bbox(r) if kind == "shape" else None,
        )
        for i, (kind, r) in enumerate(sent)
    ]


def annotate(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    client: Client, scope: Scope, fn, mapping, labels, opts: RunOptions, frames, progress
) -> None:
    """Runs fn on frames, saving every FRAMES_PER_SAVE frames in one request. With replace, a
    batch's old shapes and tags are deleted right after its new ones are saved."""
    function_names = {spec.id: spec.name for spec in fn.spec.labels}
    label_names = {label.id: label.name for label in labels}
    # Fetched once, before any write, so no batch deletes what an earlier batch saved.
    old = _old_objects(scope, frames) if opts.mode == "replace" else {}
    for start in range(0, len(frames), FRAMES_PER_SAVE):
        batch = frames[start : start + FRAMES_PER_SAVE]
        sent = []
        for frame in batch:
            image = PIL.Image.open(scope.task.get_frame(frame, quality="original"))
            # shortcut: built-in functions ignore frame_name; upgrade to the file name from the
            # task meta if user-written functions are ever allowed
            output = fn.detect(Context(str(frame), opts.conf_threshold, opts.polygons), image)
            for obj in output:  # before remap rewrites label_id; an unknown id is remap's error
                if obj.label_id in mapping and mapping[obj.label_id] is None:
                    progress.skipped[function_names[obj.label_id]] += 1
            sent += remap(output, mapping, frame)
        ids = create_objects(client, scope, sent) if sent else []
        progress.frames_done += len(batch)
        progress.added.update(label_names[r.label_id] for _, r in sent)
        if progress.keep_objects:
            progress.objects += _objects(sent, ids, label_names)
        if stale := [item for frame in batch for item in old.get(frame, [])]:
            saved = (
                f"object ids {', '.join(map(str, ids)) or 'none'}"
                if progress.keep_objects
                else f"frames {batch[0]} to {batch[-1]}"
            )
            _delete_stale(client, scope, stale, saved)


def _saved(progress: Progress, frames: list[int], opts: RunOptions) -> str:
    """What a failed run left in CVAT, and what running it again does."""
    if not progress.frames_done:
        return "nothing was saved"
    again = "replaces them again" if opts.mode == "replace" else "duplicates them"
    return (
        f"the first {progress.frames_done} of {len(frames)} frames (through frame "
        f"{frames[progress.frames_done - 1]}) are saved; running it again {again}"
    )


# One model run at a time: the CPU is the bottleneck, and two runs would also double the memory.
_LOCK = threading.Lock()
_ACTIVE: str | None = None  # the background run holding the lock; None for a frame run


# REVIEW: we should use context manager instead. It's more safe and ensure lock releasing.
def _acquire(run_id: str | None = None) -> None:
    global _ACTIVE  # pylint: disable=global-statement
    if not _LOCK.acquire(blocking=False):
        if _ACTIVE is None:
            raise InvalidArgument("another auto_annotate call is in progress; try again shortly")
        raise InvalidArgument(
            f"background run {_ACTIVE} is in progress; follow it with "
            f"get_auto_annotation_run(run_id={_ACTIVE!r}) and try again when it ends"
        )
    _ACTIVE = run_id


def _release() -> None:
    global _ACTIVE  # pylint: disable=global-statement
    _ACTIVE = None
    _LOCK.release()


def resolve_scope(client: Client, task_id: int) -> Scope:
    scope = target_scope(client, task_id, None)
    if str(scope.task.dimension) == "3d":
        raise InvalidArgument(
            f"task {scope.task.id} is a 3D task; auto-annotation supports 2D images"
        )
    check_editable(scope)
    return scope


def _answer(task_id: int, opts: RunOptions, started: float, **rest) -> AutoAnnotation:
    return AutoAnnotation(
        task_id=task_id,
        function=opts.function,
        model=opts.model,
        mode=opts.mode,
        seconds=round(time.monotonic() - started, 1),
        **rest,
    )


def _note(objects: list, progress: Progress, labels) -> str | None:
    if objects:
        return None
    names = ", ".join(sorted(label.name for label in labels))
    if progress.skipped:
        return f"nothing matched: the function's labels are not among the task's ({names})"
    return "the model found nothing above conf_threshold on this frame"


def auto_annotate_frame(
    client: Client, task_id: int, frame: int, opts: RunOptions
) -> AutoAnnotation:
    started = time.monotonic()
    scope = resolve_scope(client, task_id)
    scope.check_frame(frame)
    if frame not in scope.frames():
        raise InvalidArgument(f"frame {frame} is deleted in task {scope.task.id}")
    labels = scope.task.get_labels()
    progress = Progress()
    _acquire()
    try:
        fn = load_function(opts)
        mapping = label_map(fn, labels, opts.polygons)
        annotate(client, scope, fn, mapping, labels, opts, [frame], progress)
    finally:
        _release()
    return _answer(
        task_id,
        opts,
        started,
        frame=frame,
        objects=progress.objects,
        skipped=dict(progress.skipped),
        note=_note(progress.objects, progress, labels),
    )


class AutoAnnotationRuns(BaseModel):
    runs: list[AutoAnnotationRun]
    note: str = (
        "runs are kept in memory since the server started; a restart forgets them, and a run in "
        f"progress stops (it saves every {FRAMES_PER_SAVE} frames, and those stay in CVAT)"
    )


# REVIEW: definitely it's not a production ready decision to remember all runs
# shortcut: runs live in memory and are never pruned (a few hundred bytes each); persist them
# if the server ever needs to resume runs after a restart.
_RUNS: dict[str, tuple[AutoAnnotationRun, Progress]] = {}
_RUN_NUMBERS = itertools.count(1)


def _snapshot(run_id: str) -> AutoAnnotationRun:
    run, progress = _RUNS[run_id]
    return run.model_copy(
        update={
            "frames_done": progress.frames_done,
            "added": dict(progress.added),
            "skipped": dict(progress.skipped),
        }
    )


def _run(  # pylint: disable=too-many-arguments,too-many-positional-arguments  # thread args
    run: AutoAnnotationRun,
    progress: Progress,
    client_factory,
    opts: RunOptions,
    fn,
    mapping,
    labels,
    frames,
):
    client = None
    try:
        # REVIEW: let's use context manager instead
        # Its own client: the request's client may close when the MCP session ends.
        client = client_factory()
        scope = Scope(client.tasks.retrieve(run.task_id))
        annotate(client, scope, fn, mapping, labels, opts, frames, progress)
        run.state = "finished"
    except Exception as exc:  # pylint: disable=broad-exception-caught  # recorded for the model
        logger.exception("auto-annotation run %s failed", run.run_id)
        run.state = "failed"
        error = explain(exc) or f"{type(exc).__name__}: {exc}"
        run.error = f"{error}; {_saved(progress, frames, opts)}"
    finally:
        run.finished = datetime.now(timezone.utc)
        _release()  # first: a failing close below must not keep every later call locked out
        if client is not None:
            client.close()


def start_run(
    client: Client,
    client_factory: Callable[[], Client],
    task_id: int,
    opts: RunOptions,
) -> AutoAnnotation:
    """Check everything here, then annotate the whole task in a background thread."""
    started = time.monotonic()
    scope = resolve_scope(client, task_id)
    task = scope.task
    frames = scope.frames()
    if not frames:
        raise InvalidArgument(f"task {task.id} has no frames to annotate")
    labels = task.get_labels()
    run_id = f"run-{next(_RUN_NUMBERS)}"
    _acquire(run_id)
    try:
        fn = load_function(opts)  # in this thread, so a bad model or download fails the call
        mapping = label_map(fn, labels, opts.polygons)  # and so does a model that doesn't fit
    except BaseException:
        _release()
        raise
    run = AutoAnnotationRun(
        run_id=run_id,
        state="running",
        task_id=task.id,
        function=opts.function,
        model=opts.model,
        mode=opts.mode,
        frames_total=len(frames),
        started=datetime.now(timezone.utc),
    )
    progress = Progress(keep_objects=False)
    _RUNS[run_id] = (run, progress)
    answer = _snapshot(run_id)  # before the thread can change it
    threading.Thread(
        target=_run,
        args=(run, progress, client_factory, opts, fn, mapping, labels, frames),
        name=f"auto-annotation {run_id}",
        daemon=True,
    ).start()
    return _answer(
        task.id,
        opts,
        started,
        note=f"running in the background; follow it with get_auto_annotation_run(run_id={run_id!r})",
        run=answer,
    )


def get_runs(run_id: str | None) -> AutoAnnotationRuns:
    if run_id is None:
        # a copy: another request's start_run may add a run meanwhile
        return AutoAnnotationRuns(runs=[_snapshot(r) for r in reversed(list(_RUNS))])
    if run_id not in _RUNS:
        known = ", ".join(_RUNS) or "none since the server started"
        raise InvalidArgument(f"unknown run_id {run_id!r}; known runs: {known}")
    return AutoAnnotationRuns(runs=[_snapshot(run_id)])
