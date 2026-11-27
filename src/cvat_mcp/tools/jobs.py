from typing import Annotated

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.resolve import Resolve
from pydantic import Field

from cvat_mcp.confirm import YES, Answer, ask, require_yes
from cvat_mcp.server import get_client, tool
from cvat_mcp.services import job_edits
from cvat_mcp.services.job_edits import DeletedJob, GtJob, Stage, State
from cvat_mcp.services.tasks import JobInfo


@tool(write=True)
def update_job(
    ctx: Context,
    job_id: int,
    assignee: str | None = None,
    stage: Stage | None = None,
    state: State | None = None,
) -> JobInfo:
    """Use to assign a job (a CVAT username; "" unassigns) or move it through the workflow
    (stage annotation -> validation -> acceptance; state new, in progress, rejected,
    completed), only when the user asks. Give only what changes. Changing the stage without a
    state resets the state to "new" (CVAT's rule); the result shows the job as it is now."""
    return job_edits.update_job(get_client(ctx), job_id, assignee, stage, state)


@tool(write=True)
def create_gt_job(
    ctx: Context,
    task_id: int,
    frame_count: int | None = None,
    frames: Annotated[list[int], Field(min_length=1, max_length=10000)] | None = None,
    seed: int | None = None,
) -> GtJob:
    """Use to add a ground-truth (validation) job to a task, only when the user asks: either
    frame_count random frames (seed makes the choice repeatable) or the given task frame numbers.
    CVAT allows one per task and only for image tasks. Fill it with import_annotations (e.g. a
    COCO file) or the annotation tools with its job_id. Returns the job id and its
    task frame numbers (frames null: the job was created but its frames could not be read; do not
    retry)."""
    return job_edits.create_gt_job(get_client(ctx), task_id, frame_count, frames, seed)


def _confirm_delete_gt_job(ctx: Context, job_id: int):
    return ask(
        ctx,
        "delete_gt_job",
        {"job_id": job_id},
        f"Delete ground-truth job {job_id} with its annotations? This can't be undone.",
    )


@tool(write=True, destructive_hint=True, idempotent_hint=True)
def delete_gt_job(
    ctx: Context,
    job_id: int,
    confirmation: Annotated[Answer, Resolve(_confirm_delete_gt_job)] = YES,
) -> DeletedJob:
    """Use to delete a ground-truth job with its annotations, only when the user explicitly asks.
    Only in tasks owned by this server's CVAT account; annotation jobs can't be deleted (CVAT
    makes them from the task's segments). The client may ask the user to confirm first; "the user
    declined" means it was not done: don't call again unless the user asks."""
    require_yes(confirmation, f"ground-truth job {job_id} was not deleted")
    return job_edits.delete_gt_job(get_client(ctx), job_id)
