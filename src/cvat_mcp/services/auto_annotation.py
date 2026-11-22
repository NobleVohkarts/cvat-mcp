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

# Private on purpose: annotate_task() runs whole tasks only and saves at the end; these two do
# the per-frame work. Safe while cvat-sdk is pinned (==2.83.0); tests cover the mapping.
from cvat_sdk.auto_annotation.driver import _AnnotationMapper, _DetectionFunctionContextImpl
from cvat_sdk.auto_annotation.exceptions import BadFunctionError
from pydantic import BaseModel

from cvat_mcp.errors import InvalidArgument, explain
from cvat_mcp.services.annotation_edits import (
    REQUEST,
    check_editable,
    object_key,
    patch_annotations,
)
from cvat_mcp.services.annotations import Scope, scope_of, shape_bbox

Function = Literal["detection", "instance_segmentation", "keypoint_detection", "classification"]
Mode = Literal["append", "replace"]

MAX_FRAMES = 20
DEFAULT_MODELS: dict[str, str] = {
    "detection": "fasterrcnn_resnet50_fpn_v2",
    "instance_segmentation": "maskrcnn_resnet50_fpn_v2",
    "keypoint_detection": "keypointrcnn_resnet50_fpn",
    "classification": "resnet50",
}
_MODULES = {f: f"cvat_sdk.auto_annotation.functions.torchvision_{f}" for f in DEFAULT_MODELS}
# Detection models that also fit the segmentation / keypoint functions.
_MODEL_PART = {"instance_segmentation": "maskrcnn", "keypoint_detection": "keypointrcnn"}


@dataclass(frozen=True)
class RunOptions:
    function: Function
    model: str
    conf_threshold: float
    polygons: bool
    mode: Mode


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
    job_id: int | None
    function: str
    model: str
    mode: str
    frames_total: int
    frames_done: int
    added: dict[str, int]
    skipped: dict[str, int]
    started: datetime
    finished: datetime | None = None
    error: str | None = None


class AutoAnnotation(BaseModel):
    task_id: int
    job_id: int | None
    function: str
    model: str
    mode: str
    frames: list[int]
    objects: list[AutoObject]
    skipped: dict[str, int]  # the function's label -> detections dropped (no such task label)
    seconds: float
    note: str | None = None
    run: AutoAnnotationRun | None = None  # set for a background run (no frames given)


@dataclass
class Progress:
    frames_done: int = 0
    added: Counter = field(default_factory=Counter)
    skipped: Counter = field(default_factory=Counter)
    objects: list[AutoObject] = field(default_factory=list)
    saved_frames: list[int] = field(default_factory=list)


def check_options(
    function: Function, model: str | None, conf_threshold: float, polygons: bool, mode: Mode
) -> RunOptions:
    if function not in DEFAULT_MODELS:
        raise InvalidArgument(f"function must be one of: {', '.join(DEFAULT_MODELS)}")
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
    fn = importlib.import_module(_MODULES[opts.function]).create(opts.model)
    _loaded = (key, fn)
    return fn


def target_frames(scope: Scope, frames: list[int] | None) -> list[int]:
    allowed = scope.frames()
    if frames is None:
        return allowed
    if not 1 <= len(frames) <= MAX_FRAMES:
        raise InvalidArgument(
            f"pass 1 to at most {MAX_FRAMES} frames, or none to annotate all frames in the "
            "background"
        )
    if len(set(frames)) != len(frames):
        raise InvalidArgument("a frame is listed twice")
    deleted = set(allowed)
    for frame in frames:
        scope.check_frame(frame)
        if frame not in deleted:
            raise InvalidArgument(f"frame {frame} is deleted in task {scope.task.id}")
    return sorted(frames)


def clear_frames(client: Client, scope: Scope, frames: list[int]) -> None:
    """Delete the scope's shapes and tags on those frames by explicit ids; tracks stay."""
    wanted = set(frames)
    annotations = scope.source.get_annotations()
    sent = [
        (kind, REQUEST[kind](**obj.to_dict()))
        for kind, objs in (("shape", annotations.shapes), ("tag", annotations.tags))
        for obj in objs
        if obj.frame in wanted
    ]
    if sent:
        patch_annotations(client, "delete", scope, sent)


def _bad_function(opts: RunOptions, exc: BadFunctionError) -> InvalidArgument:
    return InvalidArgument(
        f"the {opts.function} output does not fit this task's labels: {exc}. The task's label "
        "types must fit what the function outputs (e.g. a skeleton label for "
        "keypoint_detection, or type any)"
    )


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


def annotate(
    client: Client,
    scope: Scope,
    labels,
    fn,
    opts: RunOptions,
    frames: list[int],
    chunk_size: int,
    progress: Progress,
) -> None:
    """Run fn on frames, saving each chunk of chunk_size frames in one upload."""
    function_names = {label.id: label.name for label in fn.spec.labels}
    names = {label.id: label.name for label in labels}
    try:
        mapper = _AnnotationMapper(
            client.logger,
            fn.spec.labels,
            labels,
            allow_unmatched_labels=True,
            conv_mask_to_poly=opts.polygons,
        )
    except BadFunctionError as exc:
        raise _bad_function(opts, exc) from exc

    for start in range(0, len(frames), chunk_size):
        chunk = frames[start : start + chunk_size]
        sent, skipped = [], Counter()
        for frame in chunk:
            image = PIL.Image.open(scope.task.get_frame(frame, quality="original"))
            # https://github.com/pylint-dev/pylint/issues/9013 (the SDK silences it the same way)
            # pylint: disable-next=abstract-class-instantiated
            context = _DetectionFunctionContextImpl(
                # shortcut: built-in functions ignore frame_name; upgrade to the file name from
                # the task meta if user-written functions are ever allowed
                frame_name=str(frame),
                conf_threshold=opts.conf_threshold,
                conv_mask_to_poly=opts.polygons,
            )
            output = fn.detect(context, image)
            skipped.update(
                function_names[a.label_id]
                for a in output
                if function_names.get(a.label_id) not in names.values()
            )
            try:
                tags, shapes = mapper.validate_and_remap(output, frame)
            except BadFunctionError as exc:
                raise _bad_function(opts, exc) from exc
            sent += [("shape", s) for s in shapes] + [("tag", t) for t in tags]

        ids = []
        if sent:
            data = patch_annotations(client, "create", scope, sent)
            # Pair ids by content, as create_annotations does.
            new_ids = defaultdict(list)
            for kind, created in (("shape", data.shapes), ("tag", data.tags)):
                for c in created:
                    new_ids[object_key(c, kind)].append(c.id)
            ids = [(new_ids[object_key(r, kind)] or [None]).pop(0) for kind, r in sent]

        progress.objects += _objects(sent, ids, names)
        progress.added.update(names[r.label_id] for _, r in sent)
        progress.skipped.update(skipped)
        progress.frames_done += len(chunk)
        progress.saved_frames += chunk


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


def resolve_scope(client: Client, task_id: int | None, job_id: int | None) -> Scope:
    if (task_id is None) == (job_id is None):
        raise InvalidArgument("pass exactly one of task_id or job_id")
    if task_id is None:
        task_id = client.jobs.retrieve(job_id).task_id
    task = client.tasks.retrieve(task_id)
    if str(task.dimension) == "3d":
        raise InvalidArgument(f"task {task.id} is a 3D task; auto-annotation supports 2D images")
    scope = scope_of(client, task, job_id)
    check_editable(scope)
    return scope


def _note(progress: Progress, labels) -> str | None:
    if progress.objects:
        return None
    names = ", ".join(sorted(label.name for label in labels))
    if progress.skipped:
        return f"nothing matched: the function's labels are not among the task's ({names})"
    return "the model found nothing above conf_threshold on these frames"


def auto_annotate_frames(
    client: Client,
    task_id: int | None,
    job_id: int | None,
    frames: list[int],
    opts: RunOptions,
    chunk_size: int,
) -> AutoAnnotation:
    started = time.monotonic()
    scope = resolve_scope(client, task_id, job_id)
    frames = target_frames(scope, frames)
    labels = scope.task.get_labels()
    progress = Progress()
    _acquire()
    try:
        fn = load_function(opts)
        if opts.mode == "replace":
            clear_frames(client, scope, frames)
        annotate(client, scope, labels, fn, opts, frames, chunk_size, progress)
    except Exception as exc:
        if not progress.saved_frames:
            raise
        # The model would otherwise retry the whole call and duplicate the saved objects.
        ids = ", ".join(str(o.id) for o in progress.objects)
        raise InvalidArgument(
            f"auto-annotation failed after saving frames "
            f"{', '.join(map(str, progress.saved_frames))} (object ids: {ids or 'none'}): "
            f"{explain(exc) or exc}. Those frames are done: do not repeat them."
        ) from exc
    finally:
        _release()
    return AutoAnnotation(
        task_id=scope.task.id,
        job_id=scope.job_id,
        function=opts.function,
        model=opts.model,
        mode=opts.mode,
        frames=frames,
        objects=progress.objects,
        skipped=dict(progress.skipped),
        seconds=round(time.monotonic() - started, 1),
        note=_note(progress, labels),
    )


class AutoAnnotationRuns(BaseModel):
    runs: list[AutoAnnotationRun]
    note: str = (
        "runs are kept in memory since the server started; a restart forgets them, and a run in "
        "progress stops (the frames it saved stay in CVAT)"
    )


logger = logging.getLogger(__name__)

# shortcut: runs live in memory and are never pruned (a few hundred bytes each); persist them
# if the server ever needs to resume runs after a restart.
_RUNS: dict[str, tuple[AutoAnnotationRun, Progress]] = {}
_THREADS: dict[str, threading.Thread] = {}
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


def _run(run: AutoAnnotationRun, progress: Progress, client_factory, opts, fn, frames, size):
    client = None
    try:
        # Its own client: the request's client may close when the MCP session ends.
        client = client_factory()
        task = client.tasks.retrieve(run.task_id)
        scope = scope_of(client, task, run.job_id)
        if opts.mode == "replace":
            clear_frames(client, scope, frames)
        annotate(client, scope, task.get_labels(), fn, opts, frames, size, progress)
        run.state = "finished"
    except Exception as exc:  # pylint: disable=broad-exception-caught  # recorded for the model
        logger.exception("auto-annotation run %s failed", run.run_id)
        run.state = "failed"
        run.error = explain(exc) or f"{type(exc).__name__}: {exc}"
    finally:
        run.finished = datetime.now(timezone.utc)
        if client is not None:
            client.close()
        _release()


def start_run(
    client: Client,
    client_factory: Callable[[], Client],
    task_id: int | None,
    job_id: int | None,
    opts: RunOptions,
    chunk_size: int,
) -> AutoAnnotation:
    """Check everything here, then annotate all the target's frames in a background thread."""
    started = time.monotonic()
    scope = resolve_scope(client, task_id, job_id)
    frames = target_frames(scope, None)
    if not frames:
        raise InvalidArgument(f"task {scope.task.id} has no frames to annotate")
    run_id = f"run-{next(_RUN_NUMBERS)}"
    _acquire(run_id)
    try:
        fn = load_function(opts)  # in this thread, so a bad model or download fails the call
    except BaseException:
        _release()
        raise
    run = AutoAnnotationRun(
        run_id=run_id,
        state="running",
        task_id=scope.task.id,
        job_id=scope.job_id,
        function=opts.function,
        model=opts.model,
        mode=opts.mode,
        frames_total=len(frames),
        frames_done=0,
        added={},
        skipped={},
        started=datetime.now(timezone.utc),
    )
    progress = Progress()
    _RUNS[run_id] = (run, progress)
    answer = _snapshot(run_id)  # before the thread can change it
    thread = threading.Thread(
        target=_run,
        args=(run, progress, client_factory, opts, fn, frames, chunk_size),
        name=f"auto-annotation {run_id}",
        daemon=True,
    )
    _THREADS[run_id] = thread
    thread.start()
    return AutoAnnotation(
        task_id=scope.task.id,
        job_id=scope.job_id,
        function=opts.function,
        model=opts.model,
        mode=opts.mode,
        frames=[],
        objects=[],
        skipped={},
        seconds=round(time.monotonic() - started, 1),
        note=f"running in the background; follow it with get_auto_annotation_run(run_id={run_id!r})",
        run=answer,
    )


def get_runs(run_id: str | None) -> AutoAnnotationRuns:
    if run_id is None:
        return AutoAnnotationRuns(runs=[_snapshot(r) for r in reversed(_RUNS)])
    if run_id not in _RUNS:
        known = ", ".join(_RUNS) or "none since the server started"
        raise InvalidArgument(f"unknown run_id {run_id!r}; known runs: {known}")
    return AutoAnnotationRuns(runs=[_snapshot(run_id)])
