import json
import logging
import time
from pathlib import Path
from typing import Literal

from cvat_sdk import Client
from cvat_sdk.api_client.exceptions import ApiException
from cvat_sdk.core.exceptions import BackgroundRequestException
from pydantic import BaseModel

from cvat_mcp.config import resolve_in_workdir
from cvat_mcp.errors import InvalidArgument, explain
from cvat_mcp.services.cloud_storages import cloud_source, parse_bucket_url, resolve_storage
from cvat_mcp.services.exports import check_format

logger = logging.getLogger("cvat_mcp")

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
    if not path.exists():
        raise InvalidArgument(f"no such file in the work directory: {file!r}")
    if not path.is_file():
        raise InvalidArgument(f"{file!r} is not a file")
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
    job = client.jobs.retrieve(job_id) if job_id is not None else None
    task = client.tasks.retrieve(job.task_id if job else task_id)
    check_format(client, format_name, str(task.dimension), "import")
    target = job or task
    path = _local_file(workdir, file) if file is not None else None
    storage_id, registered = resolve_storage(client, ref) if ref else (None, False)  # may register

    started = time.monotonic()
    try:
        if path is not None:
            target.import_annotations(format_name, path, import_mode=mode, status_check_period=1)
        else:
            api = client.api_client.jobs_api if job else client.api_client.tasks_api
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
            client.wait_for_completion(rq_id, status_check_period=1)
    except BaseException as exc:
        if registered:  # a storage registered only for this import is no use without it
            try:
                client.api_client.cloudstorages_api.destroy(storage_id)
            except Exception:  # pylint: disable=broad-exception-caught
                logger.exception("could not remove cloud storage %s", storage_id)
        refused = isinstance(exc, BackgroundRequestException) or (
            isinstance(exc, ApiException) and exc.status == 400
        )
        if not refused:
            raise
        hint = "check that the file's image file names match the task's frame names and that "
        hint += "format fits the file"
        if storage_id is not None and not registered:
            hint += (
                f", and that cloud storage #{storage_id} can read the bucket (its keys: "
                "update_cloud_storage)"
            )
        raise InvalidArgument(f"{explain(exc).rstrip()}; {hint}") from exc
    seconds = round(time.monotonic() - started, 1)

    annotations = target.get_annotations()
    counts = {
        "shapes": len(annotations.shapes),
        "tracks": len(annotations.tracks),
        "tags": len(annotations.tags),
    }
    return ImportedAnnotations(
        task_id=task.id,
        job_id=job_id,
        format=format_name,
        mode=mode,
        source=file if file is not None else cloud_url,
        seconds=seconds,
        note=None if any(counts.values()) else NO_OBJECTS,
        cloud_storage_id=storage_id,
        cloud_storage_registered=registered,
        **counts,
    )
