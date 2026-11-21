import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from cvat_sdk import Client
from cvat_sdk.api_client import models
from cvat_sdk.core.proxies.tasks import ResourceType
from cvat_sdk.attributes import attribute_value_validator
from pydantic import BaseModel, ConfigDict, Field

from cvat_mcp.config import resolve_in_workdir
from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.issues import job_url
from cvat_mcp.services.tasks import JobInfo, job_info

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
        valid = attribute_value_validator(request)  # raises for bad number/select specs
    except ValueError as exc:
        raise InvalidArgument(f"attribute {spec.name!r}: {exc}") from exc
    if spec.input_type == "checkbox" and not valid(request.values[0]):
        raise InvalidArgument(f"attribute {spec.name!r}: a checkbox default is true or false")
    return request


def label_requests(
    specs: list[LabelSpec], existing: Iterable[str] = ()
) -> list[models.PatchedLabelRequest]:
    """CVAT label requests for new labels; names must be new to the task and unique."""
    if not specs:
        raise InvalidArgument("pass at least one label")
    seen = set(existing)
    requests = []
    for spec in specs:
        if not spec.name.strip():
            raise InvalidArgument("a label name must not be empty")
        if spec.name in seen:
            already = spec.name in set(existing)
            raise InvalidArgument(
                f"the task already has a label {spec.name!r}"
                if already
                else f"label {spec.name!r} is listed twice"
            )
        seen.add(spec.name)
        requests.append(
            models.PatchedLabelRequest(
                name=spec.name,
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


def resolve_files(workdir: Path, files: list[str]) -> list[Path]:
    """Files inside the workdir; a directory stands for its files, recursively and sorted."""
    resolved = []
    for name in files:
        path = resolve_in_workdir(workdir, name)
        if path.is_dir():
            found = sorted(p for p in path.rglob("*") if p.is_file())
            if not found:
                raise InvalidArgument(f"directory {name!r} has no files")
            resolved += found
        elif path.is_file():
            resolved.append(path)
        else:
            raise InvalidArgument(f"no such file in the work directory: {name!r}")
    return resolved


def _check_urls(urls: list[str]) -> None:
    for url in urls:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise InvalidArgument(f"{url!r} is not an http(s) URL")


def create_task(
    client: Client,
    base_url: str,
    workdir: Path,
    name: str,
    labels: list[LabelSpec] | None,
    project_id: int | None,
    files: list[str] | None,
    urls: list[str] | None,
    segment_size: int | None,
) -> CreatedTask:
    # Argument checks first: they need no requests.
    if not name.strip():
        raise InvalidArgument("the task name must not be empty")
    if (labels is None) == (project_id is None):
        raise InvalidArgument(
            "pass exactly one of labels or project_id (a project's tasks use its labels)"
        )
    if (files is None) == (urls is None):
        raise InvalidArgument("pass exactly one of files or urls")
    if files is not None:
        resources, resource_type = resolve_files(workdir, files), ResourceType.LOCAL
    else:
        _check_urls(urls)
        resources, resource_type = list(urls), ResourceType.REMOTE
    spec = {"name": name, "project_id": project_id, "segment_size": segment_size}
    if labels is not None:
        spec["labels"] = label_requests(labels)

    task = client.tasks.create(
        models.TaskWriteRequest(**{k: v for k, v in spec.items() if v is not None})
    )
    # Two steps instead of create_from_data(), which loses the task when the upload fails.
    try:
        task.upload_data(
            resources=resources,
            resource_type=resource_type,
            params={"sorting_method": "natural"},
            wait_for_completion=True,
        )
    except BaseException:
        try:
            task.remove()  # an empty task is no use to anyone
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("could not remove task %s after its upload failed", task.id)
        raise
    task.fetch()
    return CreatedTask(
        task_id=task.id,
        name=task.name,
        frame_count=task.size,
        jobs=[job_info(j) for j in sorted(task.get_jobs(), key=lambda j: j.start_frame)],
        url=job_url(base_url, task.id),
    )
