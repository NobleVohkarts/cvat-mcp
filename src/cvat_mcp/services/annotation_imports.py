import json
import time
from typing import Literal

from cvat_sdk import Client
from cvat_sdk.api_client.exceptions import ApiException
from pydantic import BaseModel

from cvat_mcp.errors import InvalidArgument, explain, is_refusal
from cvat_mcp.services.cloud_storages import check_storage, cloud_source, keys_hint
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
    cloud_path: str
    cloud_storage_id: int
    shapes: int
    tracks: int
    tags: int
    seconds: float
    note: str | None = None


def _object_key(cloud_path: str) -> str:
    """The object key; it must name one object, not a folder or a glob."""
    resources, pattern = cloud_source(cloud_path)
    if pattern or resources[0].endswith("/"):  # "street" is the folder "street/"
        raise InvalidArgument(
            "cloud_path must name one annotation file (e.g. annotations/labels.json), "
            "not a folder or a glob"
        )
    return resources[0]


def import_annotations(
    client: Client,
    job_id: int | None,
    task_id: int | None,
    cloud_storage_id: int,
    cloud_path: str,
    format_name: str,
    mode: Mode,
) -> ImportedAnnotations:
    if (job_id is None) == (task_id is None):
        raise InvalidArgument("pass exactly one of job_id or task_id")
    key = _object_key(cloud_path)
    check_storage(client, cloud_storage_id, key)
    # a job knows its task and dimension: no task request for a job target
    target = client.jobs.retrieve(job_id) if job_id is not None else client.tasks.retrieve(task_id)
    check_format(client, format_name, str(target.dimension), "import")

    started = time.monotonic()
    try:
        api = client.api_client.jobs_api if job_id is not None else client.api_client.tasks_api
        # CVAT reads the object through the storage: the bucket keys never come here
        _, response = api.create_annotations(
            target.id,
            format=format_name,
            location="cloud_storage",
            cloud_storage_id=cloud_storage_id,
            filename=key,
            import_mode=mode,
            _parse_response=False,
        )
        rq_id = json.loads(response.data)["rq_id"]  # as the SDK's AnnotationUploader reads it
        client.wait_for_completion(rq_id, status_check_period=1)
    except Exception as exc:
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
        hint += "format fits the file" + keys_hint(cloud_storage_id)
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
        cloud_path=cloud_path,
        seconds=seconds,
        note=None if any(counts.values()) else NO_OBJECTS,
        cloud_storage_id=cloud_storage_id,
        **counts,
    )
