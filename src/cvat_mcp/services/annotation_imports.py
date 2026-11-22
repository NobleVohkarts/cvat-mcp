import time
from pathlib import Path
from typing import Literal

from cvat_sdk import Client
from pydantic import BaseModel

from cvat_mcp.config import resolve_in_workdir
from cvat_mcp.errors import InvalidArgument
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
    if not path.exists():
        raise InvalidArgument(f"no such file in the work directory: {file!r}")
    if not path.is_file():
        raise InvalidArgument(f"{file!r} is not a file")
    return path


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
    job = client.jobs.retrieve(job_id) if job_id is not None else None
    task = client.tasks.retrieve(job.task_id if job else task_id)
    check_format(client, format_name, str(task.dimension), "import")
    target = job or task

    started = time.monotonic()
    if file is not None:
        path = _local_file(workdir, file)
        target.import_annotations(format_name, path, import_mode=mode, status_check_period=1)
    else:
        raise NotImplementedError("cloud_url")
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
        **counts,
    )
