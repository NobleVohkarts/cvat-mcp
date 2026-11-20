from collections import Counter
from collections.abc import Iterable, Mapping

from cvat_sdk import Client
from pydantic import BaseModel


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


def clamp_limit(limit: int, *, low: int = 1, high: int = 100) -> int:
    return max(low, min(limit, high))


def count_objects_by_label(label_ids: Iterable[int], labels: Mapping[int, str]) -> dict[str, int]:
    counts = Counter(label_ids)
    return {name: counts.get(label_id, 0) for label_id, name in labels.items()}


def list_tasks(client: Client, project_id: int | None, search: str | None, limit: int) -> TaskList:
    # client.tasks.list() fetches every page and has no limit/total, so use one paginated call.
    filters = {
        k: v for k, v in {"project_id": project_id, "search": search}.items() if v is not None
    }
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


def get_task_overview(client: Client, task_id: int) -> TaskOverview:
    task = client.tasks.retrieve(task_id)
    labels = task.get_labels()
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
        jobs=[
            JobInfo(
                id=j.id,
                type=str(j.type),
                start_frame=j.start_frame,
                stop_frame=j.stop_frame,
                stage=str(j.stage),
                state=str(j.state),
                assignee=j.assignee.username if j.assignee else None,
            )
            for j in sorted(task.get_jobs(), key=lambda j: j.start_frame)
        ],
        objects_per_label=objects,
        total_objects=sum(objects.values()),
    )
