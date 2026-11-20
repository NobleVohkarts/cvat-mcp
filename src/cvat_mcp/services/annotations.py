from cvat_sdk import Client
from pydantic import BaseModel


class FrameObject(BaseModel):
    id: int
    label: str
    type: str
    bbox: list[float]
    attributes: dict[str, str]
    source: str
    occluded: bool


class FrameAnnotations(BaseModel):
    task_id: int
    frame: int
    width: int
    height: int
    objects: list[FrameObject]
    skipped_tracks: int


def shape_bbox(points: list[float]) -> list[float]:
    # shortcut: ignores rotation and treats ellipse points as corners; exact for axis-aligned
    # rectangles, upgrade when rotated shapes or ellipses matter.
    xs, ys = points[0::2], points[1::2]
    return [round(v, 1) for v in (min(xs), min(ys), max(xs), max(ys))]


def check_frame(frame: int, size: int) -> None:
    if not 0 <= frame < size:
        raise ValueError(f"frame {frame} is outside this task's frames 0..{size - 1}")


def _track_visible(track, frame: int) -> bool:
    keyframes = [s for s in track.shapes if s.frame <= frame]
    return bool(keyframes) and not max(keyframes, key=lambda s: s.frame).outside


def get_frame_annotations(client: Client, task_id: int, frame: int) -> FrameAnnotations:
    task = client.tasks.retrieve(task_id)
    check_frame(frame, task.size)
    labels = {label.id: label for label in task.get_labels()}
    attribute_names = {a.id: a.name for label in labels.values() for a in label.attributes}
    # shortcut: CVAT has no per-frame annotations endpoint, so this downloads the whole task;
    # fine for small tasks, fetch per job when tasks get large.
    annotations = task.get_annotations()
    meta = task.get_meta().frames[frame]
    return FrameAnnotations(
        task_id=task_id,
        frame=frame,
        width=meta.width,
        height=meta.height,
        objects=[
            FrameObject(
                id=s.id,
                label=labels[s.label_id].name,
                type=str(s.type),
                bbox=shape_bbox(s.points),
                attributes={attribute_names[a.spec_id]: a.value for a in s.attributes},
                source=str(s.source),
                occluded=s.occluded,
            )
            for s in annotations.shapes
            if s.frame == frame
        ],
        skipped_tracks=sum(_track_visible(t, frame) for t in annotations.tracks),
    )
