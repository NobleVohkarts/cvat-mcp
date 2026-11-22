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

    def __init__(self, task, job=None):
        self.task, self.job = task, job
        self._meta = self._job_frames = None

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


def _track_visible(track, frame: int) -> bool:
    keyframes = [s for s in track.shapes if s.frame <= frame]
    return bool(keyframes) and not max(keyframes, key=lambda s: s.frame).outside


def frame_annotations(scope: Scope, labels: list, frame: int) -> FrameAnnotations:
    """Annotations of one frame of an already retrieved task, or of one of its jobs."""
    scope.check_frame(frame)
    task = scope.task
    names = {label.id: label.name for label in labels}
    attribute_names = {a.id: a.name for label in labels for a in label.attributes}
    # shortcut: CVAT has no per-frame annotations endpoint, so this downloads the whole task;
    # fine for small tasks, fetch per job when tasks get large.
    annotations = scope.source.get_annotations()
    frames = scope.meta.frames
    meta = frames[frame] if len(frames) > 1 else frames[0]  # a video has one entry for all frames
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
                attributes={attribute_names[a.spec_id]: a.value for a in s.attributes},
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
                attributes={attribute_names[a.spec_id]: a.value for a in t.attributes},
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
