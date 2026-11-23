from cvat_sdk import Client
from pydantic import BaseModel

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.tasks import task_frames

POINT_TYPES = ("polygon", "polyline", "points")  # shapes whose points are x, y vertices


class FrameObject(BaseModel):
    id: int
    label: str
    type: str
    bbox: list[float]
    attributes: dict[str, str]
    source: str
    occluded: bool
    points: list[float] | None = None  # vertices of polygons, polylines and points only


class FrameTag(BaseModel):
    id: int
    label: str
    attributes: dict[str, str]
    source: str


class FrameAnnotations(BaseModel):
    task_id: int
    frame: int
    width: int
    height: int
    objects: list[FrameObject]
    tags: list[FrameTag]
    skipped_tracks: int
    job_id: int | None = None  # set when the annotations are one job's (e.g. ground truth)


def shape_bbox(shape) -> list[float]:
    """[x1, y1, x2, y2] around a shape's points."""
    # shortcut: ignores rotation and treats ellipse points as corners; exact for axis-aligned
    # rectangles, upgrade when rotated shapes or ellipses matter.
    if str(shape.type) == "mask":  # RLE counts followed by [left, top, right, bottom]
        points = shape.points[-4:]
    elif str(shape.type) == "skeleton":  # the skeleton itself has no points, its elements do
        points = [v for element in shape.elements for v in element.points]
    else:
        points = shape.points
    xs, ys = points[0::2], points[1::2]
    return [round(v, 1) for v in (min(xs), min(ys), max(xs), max(ys))]


def check_frame(frame: int, size: int) -> None:
    if not 0 <= frame < size:
        raise InvalidArgument(f"frame {frame} is outside this task's frames 0..{size - 1}")


class Scope:
    """Where annotations are read and written: the task's annotation jobs, or one of its jobs
    (a ground-truth job is reachable only this way). Frames stay task frame numbers."""

    def __init__(self, task, job=None, meta=None):
        self.task, self.job = task, job
        self._meta, self._job_frames = meta, None  # meta: the task's, when already fetched

    @property
    def source(self):
        """The object whose get_annotations() to call."""
        return self.job or self.task

    @property
    def job_id(self) -> int | None:
        return self.job.id if self.job is not None else None

    @property
    def name(self) -> str:
        return f"job {self.job.id}'s" if self.job is not None else "this task's"

    @property
    def meta(self):
        """The task's data meta, fetched once."""
        if self._meta is None:
            self._meta = self.task.get_meta()
        return self._meta

    @property
    def job_frames(self):
        """The job's task frames, fetched once (only when needed)."""
        if self._job_frames is None:
            if str(self.job.type) == "ground_truth":
                job_meta = self.job.get_meta()
                # deleted_frames (task frame numbers) include disabled validation frames
                self._job_frames = set(task_frames(self.meta, job_meta.included_frames)) - set(
                    job_meta.deleted_frames or []
                )
            else:
                self._job_frames = range(self.job.start_frame, self.job.stop_frame + 1)
        return self._job_frames

    def frames(self) -> list[int]:
        """The task's (or the job's) frames in order, without the task's deleted frames."""
        frames = range(self.task.size) if self.job is None else sorted(self.job_frames)
        deleted = set(self.meta.deleted_frames or [])
        return [f for f in frames if f not in deleted]

    def check_frame(self, frame: int) -> None:
        check_frame(frame, self.task.size)
        if self.job is None:
            return
        # CVAT doesn't check this on job writes: an object could land outside the job's frames
        if frame not in self.job_frames:
            listed = sorted(self.job_frames)
            shown = ", ".join(map(str, listed[:20])) + (", …" if len(listed) > 20 else "")
            raise InvalidArgument(
                f"frame {frame} is not in job {self.job.id} ({self.job.type}); its frames: {shown}"
            )


def scope_of(client: Client, task, job_id: int | None) -> Scope:
    """The task's scope, or that of one of its jobs (job_id)."""
    if job_id is None:
        return Scope(task)
    job = client.jobs.retrieve(job_id)
    if job.task_id != task.id:
        raise InvalidArgument(f"job {job_id} belongs to task {job.task_id}, not to task {task.id}")
    return Scope(task, job)


def target_scope(client: Client, task_id: int | None, job_id: int | None) -> Scope:
    """The scope of exactly one of a task (task_id) or a job (job_id, whose task is looked up)."""
    if (task_id is None) == (job_id is None):
        raise InvalidArgument("pass exactly one of task_id or job_id")
    job = client.jobs.retrieve(job_id) if job_id is not None else None
    return Scope(client.tasks.retrieve(job.task_id if job is not None else task_id), job)


def _names(labels: list) -> tuple[dict[int, str], dict[int, str]]:
    """Label names and attribute names by id."""
    names = {label.id: label.name for label in labels}
    return names, {a.id: a.name for label in labels for a in label.attributes}


def _attributes(obj, attribute_names: dict[int, str]) -> dict[str, str]:
    return {attribute_names[a.spec_id]: a.value for a in obj.attributes}


def _frame_meta(meta, frame: int):
    frames = meta.frames
    return frames[frame] if len(frames) > 1 else frames[0]  # a video has one entry for all frames


def _track_visible(track, frame: int) -> bool:
    keyframes = [s for s in track.shapes if s.frame <= frame]
    return bool(keyframes) and not max(keyframes, key=lambda s: s.frame).outside


def frame_annotations(scope: Scope, labels: list, frame: int) -> FrameAnnotations:
    """Annotations of one frame of an already retrieved task, or of one of its jobs."""
    scope.check_frame(frame)
    task = scope.task
    names, attribute_names = _names(labels)
    # shortcut: CVAT has no per-frame annotations endpoint, so this downloads the whole task;
    # fine for small tasks, fetch per job when tasks get large.
    annotations = scope.source.get_annotations()
    meta = _frame_meta(scope.meta, frame)
    return FrameAnnotations(
        task_id=task.id,
        frame=frame,
        width=meta.width,
        height=meta.height,
        objects=[
            FrameObject(
                id=s.id,
                label=names[s.label_id],
                type=str(s.type),
                bbox=shape_bbox(s),
                attributes=_attributes(s, attribute_names),
                source=str(s.source),
                occluded=s.occluded,
                points=list(s.points) if str(s.type) in POINT_TYPES else None,
            )
            for s in annotations.shapes
            if s.frame == frame
        ],
        tags=[
            FrameTag(
                id=t.id,
                label=names[t.label_id],
                attributes=_attributes(t, attribute_names),
                source=str(t.source),
            )
            for t in annotations.tags
            if t.frame == frame
        ],
        skipped_tracks=sum(_track_visible(t, frame) for t in annotations.tracks),
        job_id=scope.job_id,
    )


def get_frame_annotations(
    client: Client, task_id: int, frame: int, job_id: int | None = None
) -> FrameAnnotations:
    task = client.tasks.retrieve(task_id)
    return frame_annotations(scope_of(client, task, job_id), task.get_labels(), frame)


SHAPE_CAP = 2000  # shapes per call: more floods the model's context


class JobFrame(BaseModel):
    frame: int
    name: str
    width: int
    height: int


class JobShape(BaseModel):
    id: int
    frame: int
    label: str
    type: str
    points: list[float]
    rotation: float | None = None  # degrees, left out when 0
    group: int
    occluded: bool
    z_order: int
    source: str
    attributes: dict[str, str] | None = None


class JobTag(BaseModel):
    id: int
    frame: int
    label: str
    source: str
    attributes: dict[str, str] | None = None


class JobAnnotations(BaseModel):
    task_id: int
    job_id: int
    job_type: str
    frames: list[JobFrame]
    shapes: list[JobShape]
    tags: list[JobTag]
    skipped_tracks: int  # tracks are not supported yet


def job_annotations(scope: Scope, labels: list) -> JobAnnotations:
    """All annotations of the scope's job, on its frames only (deleted frames dropped)."""
    names, attribute_names = _names(labels)
    annotations = scope.job.get_annotations()
    frames = scope.frames()
    kept = set(frames)

    def frame_info(frame: int) -> JobFrame:
        meta = _frame_meta(scope.meta, frame)
        return JobFrame(frame=frame, name=meta.name, width=meta.width, height=meta.height)

    return JobAnnotations(
        task_id=scope.task.id,
        job_id=scope.job.id,
        job_type=str(scope.job.type),
        frames=[frame_info(f) for f in frames],
        shapes=[
            JobShape(
                id=s.id,
                frame=s.frame,
                label=names[s.label_id],
                type=str(s.type),
                points=list(s.points),
                rotation=s.rotation or None,
                group=s.group or 0,
                occluded=s.occluded,
                z_order=s.z_order or 0,
                source=str(s.source),
                attributes=_attributes(s, attribute_names) or None,
            )
            for s in annotations.shapes
            if s.frame in kept
        ],
        tags=[
            JobTag(
                id=t.id,
                frame=t.frame,
                label=names[t.label_id],
                source=str(t.source),
                attributes=_attributes(t, attribute_names) or None,
            )
            for t in annotations.tags
            if t.frame in kept
        ],
        skipped_tracks=len(annotations.tracks),
    )


def download_annotations(
    client: Client, task_id: int | None, job_id: int | None
) -> list[JobAnnotations]:
    """One job's annotations (job_id, any type), or those of each annotation job of a task."""
    target = target_scope(client, task_id, job_id)
    task = target.task
    if target.job is not None:
        jobs = [target.job]
    else:
        jobs = sorted(
            (j for j in task.get_jobs() if str(j.type) == "annotation"),
            key=lambda j: j.start_frame,
        )
        if not jobs:
            raise InvalidArgument(f"task {task_id} has no annotation jobs")
    labels = task.get_labels()
    results = [job_annotations(Scope(task, job, target.meta), labels) for job in jobs]
    total = sum(len(r.shapes) for r in results)
    if total > SHAPE_CAP:
        counts = ", ".join(f"job {r.job_id}: {len(r.shapes)}" for r in results)
        raise InvalidArgument(
            f"{total} shapes is over the {SHAPE_CAP}-shape limit ({counts}); pass job_id for "
            "one job, or read frame by frame with get_frame_annotations"
        )
    return results
