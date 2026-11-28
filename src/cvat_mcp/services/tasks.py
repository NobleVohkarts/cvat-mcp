from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Literal

from cvat_sdk import Client
from pydantic import BaseModel

from cvat_mcp.errors import InvalidArgument

Stage = Literal["annotation", "validation", "acceptance"]
State = Literal["new", "in progress", "rejected", "completed"]
JobType = Literal["annotation", "ground_truth", "replica"]


def job_url(base_url: str, task_id: int, job_id: int | None = None, frame: int = 0) -> str:
    """A CVAT UI link to a task, or to a frame of one of its jobs."""
    url = f"{base_url.rstrip('/')}/tasks/{task_id}"
    return f"{url}/jobs/{job_id}?frame={frame}" if job_id is not None else url


class JobProgress(BaseModel):
    total: int
    completed: int
    validation: int


class TaskSummary(BaseModel):
    id: int
    name: str
    project_id: int | None
    frame_count: int | None
    status: str
    jobs: JobProgress


class TaskList(BaseModel):
    total: int
    tasks: list[TaskSummary]


class LabelInfo(BaseModel):
    id: int
    name: str
    type: str
    attributes: list[str]


class JobInfo(BaseModel):
    id: int
    type: str
    start_frame: int
    stop_frame: int
    stage: str
    state: str
    assignee: str | None


class JobSummary(JobInfo):
    task_id: int
    task_name: str | None
    project_id: int | None
    frame_count: int
    issues: int
    url: str


class JobList(BaseModel):
    total: int
    jobs: list[JobSummary]


class TaskOverview(BaseModel):
    id: int
    name: str
    project_id: int | None
    project_name: str | None
    frame_count: int | None
    labels: list[LabelInfo]
    jobs: list[JobInfo]
    objects_per_label: dict[str, int]
    total_objects: int


def task_frames(meta, data_frames) -> list[int]:
    """Task frame numbers (0-based) for data frame numbers, as CVAT stores GT frames;
    `meta` is the task's data meta (start frame and frame filter)."""
    step = int(meta.frame_filter.split("=")[1]) if meta.frame_filter.startswith("step=") else 1
    return sorted((f - meta.start_frame) // step for f in data_frames)


def jobs_of(task) -> list[JobInfo]:
    """The task's jobs in frame order (callers pick a job by frame)."""
    return [job_info(j) for j in sorted(task.get_jobs(), key=lambda j: j.start_frame)]


def job_info(job) -> JobInfo:
    return JobInfo(
        id=job.id,
        type=str(job.type),
        start_frame=job.start_frame,
        stop_frame=job.stop_frame,
        stage=str(job.stage),
        state=str(job.state),
        assignee=job.assignee.username if job.assignee else None,
    )


def clamp_limit(limit: int, *, low: int = 1, high: int = 100) -> int:
    return max(low, min(limit, high))


def count_objects_by_label(label_ids: Iterable[int], labels: Mapping[int, str]) -> dict[str, int]:
    counts = Counter(label_ids)
    return {name: counts.get(label_id, 0) for label_id, name in labels.items()}


def org_of(client: Client, task_id: int | None, project_id: int | None) -> dict:
    """The org context of the filtered task or project: without it CVAT lists a user who isn't a
    superuser only their personal workspace, so an org's tasks and jobs would come back empty."""
    if task_id is not None:
        found, _ = client.api_client.tasks_api.retrieve(task_id)
    elif project_id is not None:
        found, _ = client.api_client.projects_api.retrieve(project_id)
    else:
        return {}
    return {"org_id": found.organization} if found.organization is not None else {}


def list_tasks(client: Client, project_id: int | None, search: str | None, limit: int) -> TaskList:
    # client.tasks.list() fetches every page and has no limit/total, so use one paginated call.
    filters = {
        k: v for k, v in {"project_id": project_id, "search": search}.items() if v is not None
    }
    filters |= org_of(client, None, project_id)
    page, _ = client.api_client.tasks_api.list(page_size=clamp_limit(limit), sort="-id", **filters)
    return TaskList(
        total=page.count,
        tasks=[
            TaskSummary(
                id=t.id,
                name=t.name,
                project_id=t.project_id,
                frame_count=t.size,
                status=str(t.status),
                jobs=JobProgress(
                    total=t.jobs.count, completed=t.jobs.completed, validation=t.jobs.validation
                ),
            )
            for t in page.results
        ],
    )


def list_jobs(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    client: Client,
    base_url: str,
    stage: Stage | None,
    state: State | None,
    assignee: str | None,
    task_id: int | None,
    project_id: int | None,
    type_: JobType | None,
    limit: int,
) -> JobList:
    """Jobs across tasks, newest first; one list request (a job carries its task's name), plus
    one for the org of a task or project filter."""
    if assignee == "":  # CVAT would filter for an empty username and silently match nothing
        raise InvalidArgument(
            "assignee must be a CVAT username; unassigned jobs can't be filtered here: list "
            "without assignee and look for assignee null"
        )
    given = {
        "stage": stage,
        "state": state,
        "assignee": assignee,
        "task_id": task_id,
        "project_id": project_id,
        "type": type_,
    }
    filters = {k: v for k, v in given.items() if v is not None} | org_of(
        client, task_id, project_id
    )
    page, _ = client.api_client.jobs_api.list(page_size=clamp_limit(limit), sort="-id", **filters)
    return JobList(
        total=page.count,
        jobs=[
            JobSummary(
                **job_info(j).model_dump(),
                task_id=j.task_id,
                task_name=j.task_name,
                project_id=j.project_id,
                frame_count=j.frame_count,
                issues=j.issues.count,
                url=job_url(base_url, j.task_id, j.id, j.start_frame),
            )
            for j in page.results
        ],
    )


def get_task_overview(client: Client, task_id: int) -> TaskOverview:
    task = client.tasks.retrieve(task_id)
    labels = task.get_labels()
    # shortcut: downloads all of the task's annotations just to count them (~24 MB for 30k
    # polygons); fine for review-sized tasks, upgrade if CVAT adds per-label counts.
    annotations = task.get_annotations()
    label_names = {label.id: label.name for label in labels}
    objects = count_objects_by_label(
        [o.label_id for o in (*annotations.shapes, *annotations.tracks, *annotations.tags)],
        label_names,
    )
    return TaskOverview(
        id=task.id,
        name=task.name,
        project_id=task.project_id,
        project_name=task.project_name,  # None when the user may not see the project
        frame_count=task.size,
        labels=[
            LabelInfo(
                id=label.id,
                name=label.name,
                type=str(label.type),
                attributes=[a.name for a in label.attributes],
            )
            for label in labels
        ],
        jobs=jobs_of(task),
        objects_per_label=objects,
        total_objects=sum(objects.values()),
    )
