import importlib
import itertools
import logging
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import PIL.Image
from cvat_sdk import Client
from cvat_sdk.api_client import models
from cvat_sdk.auto_annotation import BadFunctionError, DetectionFunctionContext, annotate_task
from cvat_sdk.core.progress import BaseProgressReporter
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

DEFAULT_MODELS: dict[str, str] = {
    "detection": "fasterrcnn_resnet50_fpn_v2",
    "instance_segmentation": "maskrcnn_resnet50_fpn_v2",
    "keypoint_detection": "keypointrcnn_resnet50_fpn",
    "classification": "resnet50",
}
# Detection models that also fit the segmentation / keypoint functions.
_MODEL_PART = {"instance_segmentation": "maskrcnn", "keypoint_detection": "keypointrcnn"}

logger = logging.getLogger(__name__)


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
    added: Counter = field(default_factory=Counter)
    skipped: Counter = field(default_factory=Counter)
    frames_done: int = 0


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


def _old_objects(scope: Scope, frames: list[int]) -> list:
    """The scope's shapes and tags on those frames, as delete requests; tracks stay."""
    wanted = set(frames)
    annotations = scope.source.get_annotations()
    return [
        (kind, REQUEST[kind](**obj.to_dict()))
        for kind, objs in (("shape", annotations.shapes), ("tag", annotations.tags))
        for obj in objs
        if obj.frame in wanted
    ]


def _delete_stale(client: Client, scope: Scope, stale: list, saved: str) -> None:
    """Deletes the old objects once the new ones are saved; a failure says the new ones are in,
    since repeating the call would save them twice."""
    try:
        patch_annotations(client, "delete", scope, stale)
    except Exception as exc:
        raise InvalidArgument(
            f"the new objects were saved ({saved}), but deleting the {len(stale)} old shapes and "
            f"tags failed: {explain(exc) or exc}. Do not repeat this call; delete the old ones "
            "with delete_annotations"
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


def _function_names(fn) -> dict[int, str]:
    return {spec.id: spec.name for spec in fn.spec.labels}


def _count(output, mapping, names: dict[int, str], progress: Progress) -> None:
    """Counts the output per function label: added if label_map matched it, else skipped."""
    for obj in output:
        if obj.label_id in names:  # an unknown id is remap's (or the SDK's) error
            counter = progress.added if mapping[obj.label_id] else progress.skipped
            counter[names[obj.label_id]] += 1


class _Counting:
    """The function as annotate_task sees it: same spec, counting what it finds."""

    def __init__(self, fn, mapping, progress: Progress):
        self.spec = fn.spec
        self._fn, self._mapping, self._progress = fn, mapping, progress
        self._names = _function_names(fn)

    def detect(self, context, image):
        output = self._fn.detect(context, image)
        _count(output, self._mapping, self._names, self._progress)
        return output


class _RunProgress(BaseProgressReporter):
    def __init__(self, progress: Progress):
        super().__init__()
        self._progress = progress

    def advance(self, delta: int):
        super().advance(delta)
        self._progress.frames_done += delta


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


# One model run at a time: the CPU is the bottleneck, and two runs would also double the memory.
_LOCK = threading.Lock()
_ACTIVE: str | None = None  # the background run holding the lock; None for a frame run


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
        image = PIL.Image.open(scope.task.get_frame(frame, quality="original"))
        # shortcut: built-in functions ignore frame_name; upgrade to the file name from the task
        # meta if user-written functions are ever allowed
        output = fn.detect(Context(str(frame), opts.conf_threshold, opts.polygons), image)
        _count(output, mapping, _function_names(fn), progress)
        sent = remap(output, mapping, frame)
        # Fetched before the create and deleted after it, so a failed create deletes nothing.
        stale = _old_objects(scope, [frame]) if opts.mode == "replace" else []
        ids = create_objects(client, scope, sent) if sent else []
        if stale:
            _delete_stale(client, scope, stale, f"object ids {', '.join(map(str, ids)) or 'none'}")
    finally:
        _release()
    objects = _objects(sent, ids, {label.id: label.name for label in labels})
    return _answer(
        task_id,
        opts,
        started,
        frame=frame,
        objects=objects,
        skipped=dict(progress.skipped),
        note=_note(objects, progress, labels),
    )


class AutoAnnotationRuns(BaseModel):
    runs: list[AutoAnnotationRun]
    note: str = (
        "runs are kept in memory since the server started; a restart forgets them, and a run in "
        "progress stops without saving anything (a run saves all its objects when it finishes)"
    )


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
    frames,
):
    client = cache = None
    try:
        # annotate_task downloads every image into the client's cache and never removes it; our
        # own upload makes the cache stale anyway, so each run gets a directory that goes with it.
        cache = tempfile.TemporaryDirectory(prefix="cvat-mcp-aa-")
        # Its own client: the request's client may close when the MCP session ends.
        client = client_factory()
        client.config.cache_dir = Path(cache.name)
        scope = Scope(client.tasks.retrieve(run.task_id))
        # Fetched before the upload and deleted only after it, so a failed run deletes nothing.
        stale = _old_objects(scope, frames) if opts.mode == "replace" else []
        try:
            annotate_task(
                client,
                run.task_id,
                _Counting(fn, mapping, progress),
                pbar=_RunProgress(progress),
                clear_existing=False,  # True would also delete the task's tracks
                allow_unmatched_labels=True,
                conf_threshold=opts.conf_threshold,
                conv_mask_to_poly=opts.polygons,
            )
        except BadFunctionError as exc:
            raise _misfit(str(exc)) from exc
        if stale:
            _delete_stale(client, scope, stale, "every frame of the run")
        run.state = "finished"
    except Exception as exc:  # pylint: disable=broad-exception-caught  # recorded for the model
        logger.exception("auto-annotation run %s failed", run.run_id)
        run.state = "failed"
        run.error = explain(exc) or f"{type(exc).__name__}: {exc}"
    finally:
        run.finished = datetime.now(timezone.utc)
        _release()  # first: a failing cleanup below must not keep every later call locked out
        if client is not None:
            client.close()
        if cache is not None:
            cache.cleanup()


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
    if str(task.data_original_chunk_type) != "imageset":
        raise InvalidArgument(
            f"task {task.id} stores a video; a whole-task run needs images (the SDK's "
            "annotate_task), pass frame to annotate one frame at a time"
        )
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
    progress = Progress()
    _RUNS[run_id] = (run, progress)
    answer = _snapshot(run_id)  # before the thread can change it
    threading.Thread(
        target=_run,
        args=(run, progress, client_factory, opts, fn, mapping, frames),
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
