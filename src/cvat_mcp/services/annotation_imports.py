import json
import time
from pathlib import Path
from typing import Literal

from cvat_sdk import Client
from cvat_sdk.api_client.exceptions import ApiException
from pydantic import BaseModel

from cvat_mcp.config import resolve_in_workdir
from cvat_mcp.errors import InvalidArgument, explain, is_refusal
from cvat_mcp.services.cloud_storages import (
    cloud_source,
    forget_storage,
    keys_hint,
    parse_bucket_url,
    resolve_storage,
)
from cvat_mcp.services.exports import check_format

Mode = Literal["append", "replace"]
DEFAULT_FORMAT = "COCO 1.0"
NO_OBJECTS = (
    "no objects after the import: the file's image file names probably don't match the task's "
    "frame names (or the file holds no annotations for these frames)"
)


class ImportedAnnotations(BaseModel):
    task_id: int
    job_id: int | None
    format: str
    mode: Mode
    source: str
    cloud_storage_id: int | None = None
    cloud_storage_registered: bool = False
    shapes: int
    tracks: int
    tags: int
    seconds: float
    note: str | None = None


def _local_file(workdir: Path, file: str) -> Path:
    path = resolve_in_workdir(workdir, file)
    if not path.is_file():
        raise InvalidArgument(f"no such file in the work directory: {file!r}")
    return path


def _object_key(cloud_url: str):
    """The bucket link split up; it must name one object, not a folder or a glob."""
    ref = parse_bucket_url(cloud_url)
    resources, pattern = cloud_source(ref.key)
    if pattern or resources[0].endswith("/"):  # "street" is the folder "street/"
        raise InvalidArgument(
            "cloud_url must name one annotation file (e.g. .../bucket/annotations/labels.json), "
            "not a folder or a glob"
        )
    return ref._replace(key=resources[0])


def import_annotations(
    client: Client,
    workdir: Path,
    job_id: int | None,
    task_id: int | None,
    cloud_url: str | None,
    file: str | None,
    format_name: str,
    mode: Mode,
) -> ImportedAnnotations:
    if (job_id is None) == (task_id is None):
        raise InvalidArgument("pass exactly one of job_id or task_id")
    if (cloud_url is None) == (file is None):
        raise InvalidArgument("pass exactly one of cloud_url or file")
    ref = _object_key(cloud_url) if cloud_url is not None else None
    path = _local_file(workdir, file) if file is not None else None
    # a job knows its task and dimension: no task request for a job target
    target = client.jobs.retrieve(job_id) if job_id is not None else client.tasks.retrieve(task_id)
    check_format(client, format_name, str(target.dimension), "import")
    storage_id, registered = resolve_storage(client, ref) if ref else (None, False)  # may register

    started = time.monotonic()
    queued = False  # CVAT has the import: it reads the file through the storage from now on
    try:
        if path is not None:
            target.import_annotations(format_name, path, import_mode=mode, status_check_period=1)
        else:
            api = client.api_client.jobs_api if job_id is not None else client.api_client.tasks_api
            # CVAT reads the object through the storage: the bucket keys never come here
            _, response = api.create_annotations(
                target.id,
                format=format_name,
                location="cloud_storage",
                cloud_storage_id=storage_id,
                filename=ref.key,
                import_mode=mode,
                _parse_response=False,
            )
            rq_id = json.loads(response.data)["rq_id"]  # as the SDK's AnnotationUploader reads it
            queued = True
            client.wait_for_completion(rq_id, status_check_period=1)
    except BaseException as exc:
        if registered and (not queued or is_refusal(exc)):  # never under a running import
            forget_storage(client, storage_id)
        if isinstance(exc, ApiException) and exc.status == 409:  # CVAT: one import at a time
            what = f"job {job_id}" if job_id is not None else f"task {task_id}"
            raise InvalidArgument(
                f"another annotation import into {what} is still running, most likely an earlier "
                "call whose answer the client stopped waiting for (a failing import takes about a "
                "minute). Don't import again in append mode, that would duplicate the objects: "
                "wait a minute, then check the target's objects (get_task_overview, or "
                "get_frame_annotations with job_id for a job)"
            ) from exc
        if not is_refusal(exc):
            raise
        hint = "check that the file's image file names match the task's frame names and that "
        hint += "format fits the file"
        if storage_id is not None and not registered:
            hint += keys_hint(storage_id)
        raise InvalidArgument(f"{explain(exc).rstrip()}; {hint}") from exc
    seconds = round(time.monotonic() - started, 1)

    annotations = target.get_annotations()
    counts = {
        "shapes": len(annotations.shapes),
        "tracks": len(annotations.tracks),
        "tags": len(annotations.tags),
    }
    return ImportedAnnotations(
        task_id=target.task_id if job_id is not None else target.id,
        job_id=job_id,
        format=format_name,
        mode=mode,
        source=file or cloud_url,
        seconds=seconds,
        note=None if any(counts.values()) else NO_OBJECTS,
        cloud_storage_id=storage_id,
        cloud_storage_registered=registered,
        **counts,
    )
