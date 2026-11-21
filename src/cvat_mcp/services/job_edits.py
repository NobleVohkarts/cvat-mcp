from typing import Literal

from cvat_sdk import Client
from cvat_sdk.api_client import models

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.task_edits import user_id
from cvat_mcp.services.tasks import JobInfo, job_info

Stage = Literal["annotation", "validation", "acceptance"]
State = Literal["new", "in progress", "rejected", "completed"]


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
