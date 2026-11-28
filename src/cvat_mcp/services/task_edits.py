import logging
from collections.abc import Iterable
from typing import Literal
from urllib.parse import urlsplit

from cvat_sdk import Client
from cvat_sdk.api_client import models
from cvat_sdk.attributes import attribute_value_validator
from cvat_sdk.core.proxies.tasks import ResourceType
from pydantic import BaseModel, ConfigDict, Field

from cvat_mcp.errors import InvalidArgument, explain, is_refusal
from cvat_mcp.services.annotation_edits import check_value
from cvat_mcp.services.cloud_storages import check_storage, cloud_source, keys_hint
from cvat_mcp.services.tasks import JobInfo, job_url, jobs_of
from cvat_mcp.services.users import require_own, user_id

logger = logging.getLogger("cvat_mcp")

DEFAULT_VALUES = {"checkbox": ["false"], "text": [""]}


class AttributeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    input_type: Literal["select", "radio", "checkbox", "number", "text"]
    values: list[str] = Field(
        [],
        description="select/radio: the allowed values; number: [min, max, step]; "
        "checkbox: [default]",
    )
    mutable: bool = Field(False, description="may change between frames of a track")


class LabelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: Literal["any", "rectangle", "polygon", "polyline", "points", "tag"] = Field(
        "any", description="the kind of object the label may be used for"
    )
    attributes: list[AttributeSpec] = []


def _attribute_request(spec: AttributeSpec) -> models.AttributeRequest:
    request = models.AttributeRequest(
        name=spec.name,
        input_type=spec.input_type,
        values=spec.values or DEFAULT_VALUES.get(spec.input_type, []),
        mutable=spec.mutable,
    )
    try:
        attribute_value_validator(request)  # raises for bad number/select specs
    except ValueError as exc:
        raise InvalidArgument(f"attribute {spec.name!r}: {exc}") from exc
    if spec.input_type == "checkbox":
        check_value(request, request.values[0])  # the default
    return request


def label_requests(
    specs: list[LabelSpec], existing: Iterable[str] = ()
) -> list[models.PatchedLabelRequest]:
    """CVAT label requests for new labels; names must be new to the task and unique."""
    if not specs:
        raise InvalidArgument("pass at least one label")
    existing, seen = set(existing), set()
    requests = []
    for spec in specs:
        name = spec.name.strip()  # "car " next to "car" would look like one label
        if not name:
            raise InvalidArgument("a label name must not be empty")
        if name in existing:
            raise InvalidArgument(f"the task already has a label {name!r}")
        if name in seen:
            raise InvalidArgument(f"label {name!r} is listed twice")
        seen.add(name)
        requests.append(
            models.PatchedLabelRequest(
                name=name,
                type=spec.type,
                attributes=[_attribute_request(a) for a in spec.attributes],
            )
        )
    return requests


class CreatedTask(BaseModel):
    task_id: int
    name: str
    frame_count: int
    jobs: list[JobInfo]
    url: str
    cloud_storage_id: int | None = None  # echoes the storage, for cloud_path


def _check_urls(urls: list[str]) -> None:
    for url in urls:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise InvalidArgument(f"{url!r} is not an http(s) URL")


def create_task(
    client: Client,
    base_url: str,
    name: str,
    labels: list[LabelSpec] | None,
    project_id: int | None,
    urls: list[str] | None,
    cloud_storage_id: int | None,
    cloud_path: str | None,
    segment_size: int | None,
) -> CreatedTask:
    # Argument checks first: they need no requests.
    if not name.strip():
        raise InvalidArgument("the task name must not be empty")
    if (labels is None) == (project_id is None):
        raise InvalidArgument(
            "pass exactly one of labels or project_id (a project's tasks use its labels)"
        )
    if (cloud_storage_id is None) != (cloud_path is None):
        raise InvalidArgument("pass cloud_storage_id and cloud_path together")
    if (urls is None) == (cloud_path is None):
        raise InvalidArgument("pass exactly one of urls or cloud_storage_id + cloud_path")
    spec = {"name": name, "project_id": project_id, "segment_size": segment_size}
    if labels is not None:
        spec["labels"] = label_requests(labels)
    params = {"sorting_method": "natural"}
    if urls is not None:
        _check_urls(urls)
        resources, resource_type = urls, ResourceType.REMOTE
    else:
        resources, pattern = cloud_source(cloud_path)
        # before the task: a wrong storage leaves nothing behind
        check_storage(client, cloud_storage_id, resources[0] if resources else pattern)
        resource_type = ResourceType.SHARE
        params |= {"cloud_storage_id": cloud_storage_id, "use_cache": True}
        if pattern:
            params["filename_pattern"] = pattern

    task = client.tasks.create(
        models.TaskWriteRequest(**{k: v for k, v in spec.items() if v is not None})
    )
    # Two steps instead of create_from_data(), which loses the task when the upload fails.
    try:
        task.upload_data(
            resources=resources,
            resource_type=resource_type,
            params=params,
            wait_for_completion=True,  # also refreshes the task (size)
            status_check_period=1,  # the default 5 s makes even a one-image task take 5 s
        )
    except BaseException as exc:
        try:
            task.remove()  # an empty task is no use to anyone
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("could not remove task %s after its upload failed", task.id)
        if cloud_storage_id is not None and is_refusal(exc):
            hint = f"check that cloud_path {cloud_path!r} names a folder or object with images"
            hint += keys_hint(cloud_storage_id)
            raise InvalidArgument(f"{explain(exc).rstrip()}; {hint}") from exc
        raise
    return CreatedTask(
        task_id=task.id,
        name=task.name,
        frame_count=task.size,
        jobs=jobs_of(task),
        url=job_url(base_url, task.id),
        cloud_storage_id=cloud_storage_id,
    )


class UpdatedTask(BaseModel):
    task_id: int
    name: str
    project_id: int | None
    assignee: str | None
    labels: list[str]


class DeletedTask(BaseModel):
    task_id: int
    name: str
    frame_count: int | None


def update_task(
    client: Client,
    task_id: int,
    name: str | None,
    assignee: str | None,
    project_id: int | None,
    add_labels: list[LabelSpec] | None,
) -> UpdatedTask:
    if name is None and assignee is None and project_id is None and not add_labels:
        raise InvalidArgument("nothing to change: pass name, assignee, project_id or add_labels")
    if name is not None and not name.strip():
        raise InvalidArgument("the task name must not be empty")
    if project_id is not None and (name is not None or assignee is not None or add_labels):
        # CVAT applies only the move when project_id is in the same request
        raise InvalidArgument("pass project_id alone; change the other fields in another call")
    changes = {k: v for k, v in {"name": name, "project_id": project_id}.items() if v is not None}
    if add_labels:
        # Labels are only added: removing one deletes its annotations.
        changes["labels"] = label_requests(add_labels)  # checks that need no requests first
    task = client.tasks.retrieve(task_id)
    if add_labels and task.project_id is not None:
        raise InvalidArgument(
            f"task {task_id} uses the labels of project {task.project_id}; add labels to the "
            "project in the CVAT UI"
        )
    if add_labels:
        label_requests(add_labels, [l.name for l in task.get_labels()])  # names new to the task
    if assignee is not None:
        changes["assignee_id"] = user_id(client, assignee)
    task.update(models.PatchedTaskWriteRequest(**changes))
    return UpdatedTask(
        task_id=task.id,
        name=task.name,
        project_id=task.project_id,
        assignee=task.assignee.username if task.assignee else None,
        labels=[l.name for l in task.get_labels()],  # what CVAT kept, not what was asked
    )


def delete_task(client: Client, task_id: int) -> DeletedTask:
    """Delete a task with everything in it, only if this server's user owns it."""
    task = client.tasks.retrieve(task_id)
    require_own(client, task, f"task #{task_id}")
    task.remove()
    return DeletedTask(task_id=task_id, name=task.name, frame_count=task.size)
