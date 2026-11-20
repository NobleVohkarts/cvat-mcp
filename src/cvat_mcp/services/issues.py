from cvat_sdk import Client
from cvat_sdk.api_client import models
from pydantic import BaseModel

from cvat_mcp.services.annotations import check_frame, get_frame_annotations


class IssueRef(BaseModel):
    issue_id: int
    job_id: int
    frame: int
    position: list[float]
    url: str


def issue_position(object_bbox: list[float] | None, bbox: list[float] | None) -> list[float]:
    if (object_bbox is None) == (bbox is None):
        raise ValueError("pass exactly one of object_id or bbox")
    if bbox is not None and not (len(bbox) == 4 and bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        raise ValueError(
            "bbox must be [x1, y1, x2, y2] in original pixels with x1 < x2 and y1 < y2"
        )
    return list(object_bbox if bbox is None else bbox)


def find_job_for_frame(jobs, frame: int):
    return next(
        (j for j in jobs if str(j.type) == "annotation" and j.start_frame <= frame <= j.stop_frame),
        None,
    )


def create_issue(
    client: Client,
    base_url: str,
    job_id: int,
    frame: int,
    message: str,
    object_id: int | None,
    bbox: list[float] | None,
) -> IssueRef:
    job = client.jobs.retrieve(job_id)
    task = client.tasks.retrieve(job.task_id)
    check_frame(frame, task.size)
    if not job.start_frame <= frame <= job.stop_frame:
        right = find_job_for_frame(task.get_jobs(), frame)
        hint = (
            f"; use job {right.id} (frames {right.start_frame}..{right.stop_frame})"
            if right
            else ""
        )
        raise ValueError(
            f"frame {frame} is not in job {job_id} (frames {job.start_frame}..{job.stop_frame})"
            + hint
        )
    object_bbox = None
    if object_id is not None:
        objects = {o.id: o for o in get_frame_annotations(client, task.id, frame).objects}
        if object_id not in objects:
            raise ValueError(
                f"object #{object_id} is not on frame {frame}; "
                "use get_frame_annotations to see the ids on this frame"
            )
        object_bbox = objects[object_id].bbox
    position = issue_position(object_bbox, bbox)
    # CVAT expects the task frame number here and does not check it against the job's range.
    issue = client.issues.create(
        models.IssueWriteRequest(frame=frame, position=position, job=job_id, message=message)
    )
    return IssueRef(
        issue_id=issue.id,
        job_id=job_id,
        frame=frame,
        position=position,
        url=f"{base_url}/tasks/{task.id}/jobs/{job_id}?frame={frame}",
    )
