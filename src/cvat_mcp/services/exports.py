import re
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

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


_reserved: set[Path] = set()
_reserve_lock = threading.Lock()


@contextmanager
def reserve_path(path: Path) -> Iterator[Path]:
    """A free name (adding -1, -2, ...) that no parallel export can take until this one ends.

    Never overwrite: the SDK replaces the target atomically, so a name picked by two
    concurrent exports would lose the first file.
    """
    with _reserve_lock:
        candidate, n = path, 0
        while candidate.exists() or candidate in _reserved:
            n += 1
            candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
        _reserved.add(candidate)
    try:
        yield candidate
    finally:
        with _reserve_lock:
            _reserved.discard(candidate)


def check_format(
    client: Client,
    format_name: str,
    dimension: str,
    direction: Literal["export", "import"] = "export",
) -> None:
    formats, _ = client.api_client.server_api.retrieve_annotation_formats()
    listed = formats.importers if direction == "import" else formats.exporters
    names = sorted(f.name for f in listed if f.enabled and str(f.dimension) == dimension)
    if format_name not in names:
        raise InvalidArgument(
            f"unknown {direction} format '{format_name}'; "
            f"valid for a {dimension} task: {', '.join(names)}"
        )


def export_dataset(
    client: Client,
    workdir: Path,
    task_id: int,
    format_name: str,
    include_images: bool,
    filename: str | None,
) -> ExportResult:
    task = client.tasks.retrieve(task_id)
    check_format(client, format_name, str(task.dimension))
    target = resolve_in_workdir(
        workdir, filename or default_filename(task_id, format_name, include_images)
    )
    if not target.name.lower().endswith(".zip"):  # CVAT always exports a zip archive
        target = target.with_name(target.name + ".zip")
    target.parent.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    with reserve_path(target) as free:
        # The SDK starts the export, waits for CVAT's background request and downloads the
        # file; LOCAL overrides a task configured to export to cloud storage.
        saved = task.export_dataset(
            format_name,
            free,
            include_images=include_images,
            location=Location.LOCAL,
            status_check_period=1,  # the SDK default (5 s) makes even a tiny export take 5 s
        )
    return ExportResult(
        path=str(saved),
        size_bytes=saved.stat().st_size,
        seconds=round(time.monotonic() - started, 1),
    )
