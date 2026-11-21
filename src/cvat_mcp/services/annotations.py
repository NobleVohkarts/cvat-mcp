from cvat_sdk import Client
from cvat_sdk.core.proxies.tasks import Task
from pydantic import BaseModel

from cvat_mcp.errors import InvalidArgument

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


def task_frames(task, data_frames) -> list[int]:
    """Task frame numbers (0-based) for data frame numbers, as CVAT stores GT frames."""
    data = task.get_meta()
    step = int(data.frame_filter.split("=")[1]) if data.frame_filter.startswith("step=") else 1
    return sorted((f - data.start_frame) // step for f in data_frames)


def job_scope(client: Client, task, job_id: int):
    """A job of the task and its frames as task frame numbers (a GT job's are sparse)."""
    job = client.jobs.retrieve(job_id)
    if job.task_id != task.id:
        raise InvalidArgument(f"job {job_id} belongs to task {job.task_id}, not to task {task.id}")
    if str(job.type) == "ground_truth":
        frames = set(task_frames(task, job.get_meta().included_frames))
    else:
        frames = set(range(job.start_frame, job.stop_frame + 1))
    return job, frames


def check_job_frame(frame: int, job, frames: set[int]) -> None:
    # CVAT doesn't check this on job writes: an object could land outside the job's frames
    if frame not in frames:
        listed = sorted(frames)
        shown = ", ".join(map(str, listed[:20])) + (", …" if len(listed) > 20 else "")
        raise InvalidArgument(
            f"frame {frame} is not in job {job.id} ({job.type}); its frames: {shown}"
        )


def _track_visible(track, frame: int) -> bool:
    keyframes = [s for s in track.shapes if s.frame <= frame]
    return bool(keyframes) and not max(keyframes, key=lambda s: s.frame).outside


def frame_annotations(task: Task, labels: list, frame: int, source=None) -> FrameAnnotations:
    """Annotations of one frame of an already retrieved task (or of `source`, one of its jobs)."""
    check_frame(frame, task.size)
    names = {label.id: label.name for label in labels}
    attribute_names = {a.id: a.name for label in labels for a in label.attributes}
    # shortcut: CVAT has no per-frame annotations endpoint, so this downloads the whole task;
    # fine for small tasks, fetch per job when tasks get large.
    annotations = (source or task).get_annotations()
    frames = task.get_meta().frames
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
        job_id=source.id if source is not None else None,
    )


def frame_source(client: Client, task, frame: int, job_id: int | None):
    """The job whose annotations to read for this frame, or None for the task's."""
    if job_id is None:
        return None
    job, frames = job_scope(client, task, job_id)
    check_job_frame(frame, job, frames)
    return job


def get_frame_annotations(
    client: Client, task_id: int, frame: int, job_id: int | None = None
) -> FrameAnnotations:
    task = client.tasks.retrieve(task_id)
    source = frame_source(client, task, frame, job_id)
    return frame_annotations(task, task.get_labels(), frame, source)
