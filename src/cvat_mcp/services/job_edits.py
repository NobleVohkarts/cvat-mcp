from typing import Literal

from cvat_sdk import Client
from cvat_sdk.api_client import models
from pydantic import BaseModel

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.annotations import check_frame
from cvat_mcp.services.issues import require_own
from cvat_mcp.services.task_edits import user_id
from cvat_mcp.services.tasks import JobInfo, job_info

Stage = Literal["annotation", "validation", "acceptance"]
State = Literal["new", "in progress", "rejected", "completed"]


class DeletedJob(BaseModel):
    job_id: int
    task_id: int
    frame_count: int


class GtJob(BaseModel):
    job_id: int
    task_id: int
    frames: list[int]


def update_job(
    client: Client, job_id: int, assignee: str | None, stage: Stage | None, state: State | None
) -> JobInfo:
    if assignee is None and stage is None and state is None:
        raise InvalidArgument("nothing to change: pass assignee, stage or state")
    changes = {k: v for k, v in {"stage": stage, "state": state}.items() if v is not None}
    if assignee is not None:
        changes["assignee"] = user_id(client, assignee)
    job = client.jobs.retrieve(job_id)
    job.update(models.PatchedJobWriteRequest(**changes))  # refreshes the job
    return job_info(job)


def create_gt_job(
    client: Client,
    task_id: int,
    frame_count: int | None,
    frames: list[int] | None,
    seed: int | None,
) -> GtJob:
    if (frame_count is None) == (frames is None):
        raise InvalidArgument("pass exactly one of frame_count (random frames) or frames")
    if frames is not None and seed is not None:
        raise InvalidArgument("seed is for random frames (frame_count), not for given frames")
    if frames is not None and len(set(frames)) != len(frames):
        raise InvalidArgument("a frame is listed twice")
    task = client.tasks.retrieve(task_id)
    if frames is not None:
        for frame in frames:
            check_frame(frame, task.size)
        spec = {"frame_selection_method": "manual", "frames": sorted(frames)}
    else:
        if not 1 <= frame_count <= task.size:
            raise InvalidArgument(f"frame_count must be in 1..{task.size} (the task's frames)")
        spec = {"frame_selection_method": "random_uniform", "frame_count": frame_count}
        if seed is not None:
            spec["random_seed"] = seed
    # The SDK's jobs repository can't create jobs, so this goes to the API directly.
    created, _ = client.api_client.jobs_api.create(
        models.JobWriteRequest(type="ground_truth", task_id=task_id, **spec)
    )
    return GtJob(
        job_id=created.id,
        task_id=task_id,
        frames=list(client.jobs.retrieve(created.id).get_meta().included_frames),
    )


def delete_gt_job(client: Client, job_id: int) -> DeletedJob:
    """Delete a ground-truth job with its annotations, only in a task this server's user owns."""
    job = client.jobs.retrieve(job_id)
    if str(job.type) != "ground_truth":
        raise InvalidArgument(
            f"job {job_id} is an {job.type} job; CVAT makes those from the task's segments, so "
            "they can't be deleted one by one (delete the task instead)"
        )
    task = client.tasks.retrieve(job.task_id)
    require_own(client, task, f"task #{task.id}", "tasks' ground-truth jobs")
    # The SDK's Job has no remove(), so this goes to the API directly.
    client.api_client.jobs_api.destroy(job_id)
    return DeletedJob(job_id=job_id, task_id=job.task_id, frame_count=job.frame_count)
