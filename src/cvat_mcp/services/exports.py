import re
import time
from pathlib import Path

from cvat_sdk import Client
from cvat_sdk.core.proxies.types import Location
from pydantic import BaseModel

from cvat_mcp.config import resolve_in_workdir
from cvat_mcp.errors import InvalidArgument


class ExportResult(BaseModel):
    path: str
    size_bytes: int
    seconds: float


def default_filename(task_id: int, format_name: str, include_images: bool) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", format_name.lower()).strip("-")
    return f"task-{task_id}-{slug}{'-images' if include_images else ''}.zip"


def unique_path(path: Path) -> Path:
    """Never overwrite: add -1, -2, ... before the suffix."""
    candidate, n = path, 0
    while candidate.exists():
        n += 1
        candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
    return candidate


def check_format(client: Client, format_name: str) -> None:
    formats, _ = client.api_client.server_api.retrieve_annotation_formats()
    names = sorted(f.name for f in formats.exporters if f.enabled)
    if format_name not in names:
        raise InvalidArgument(f"unknown export format '{format_name}'; valid: {', '.join(names)}")


def export_dataset(
    client: Client,
    workdir: Path,
    task_id: int,
    format_name: str,
    include_images: bool,
    filename: str | None,
) -> ExportResult:
    check_format(client, format_name)
    target = resolve_in_workdir(
        workdir, filename or default_filename(task_id, format_name, include_images)
    )
    target = target.with_suffix(".zip")  # CVAT always exports a zip archive
    target.parent.mkdir(parents=True, exist_ok=True)
    target = unique_path(target)

    started = time.monotonic()
    # The SDK starts the export, waits for CVAT's background request and downloads the file;
    # LOCAL overrides a task configured to export to cloud storage.
    saved = client.tasks.retrieve(task_id).export_dataset(
        format_name,
        target,
        include_images=include_images,
        location=Location.LOCAL,
        status_check_period=1,  # the SDK default (5 s) makes even a tiny export take 5 s
    )
    return ExportResult(
        path=str(saved),
        size_bytes=saved.stat().st_size,
        seconds=round(time.monotonic() - started, 1),
    )
