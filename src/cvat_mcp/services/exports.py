import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

from cvat_sdk import Client
from cvat_sdk.api_client.exceptions import ApiException
from cvat_sdk.core.proxies.types import Location
from pydantic import BaseModel

from cvat_mcp.errors import InvalidArgument, explain, is_refusal
from cvat_mcp.services.cloud_storages import check_storage, keys_hint, object_exists


class ExportResult(BaseModel):
    cloud_storage_id: int
    cloud_path: str  # the object key written
    replaced: bool  # an object was already there and was overwritten
    seconds: float


def _export_key(cloud_path: str) -> str:
    key = cloud_path.strip().lstrip("/")  # object keys never start with "/"
    if not key or key.endswith("/") or any(c in key for c in "*?"):
        raise InvalidArgument(
            "cloud_path must name the file to write, e.g. exports/task-7-coco.zip (an object "
            "key from the bucket root), not a folder or a glob"
        )
    return key if key.lower().endswith(".zip") else key + ".zip"  # CVAT exports a zip


_exporting: set[tuple[int, str]] = set()
_exporting_lock = threading.Lock()


@contextmanager
def _reserved(storage_id: int, key: str) -> Iterator[None]:
    """Hold the key from the exists check until the upload ends: CVAT replaces an object
    silently, so two exports to one key would lose the first file."""
    with _exporting_lock:
        if (storage_id, key) in _exporting:
            raise InvalidArgument(
                f"another export to {key} in cloud storage #{storage_id} is still running; "
                "wait for it or pick another cloud_path"
            )
        _exporting.add((storage_id, key))
    try:
        yield
    finally:
        with _exporting_lock:
            _exporting.discard((storage_id, key))


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


def _exists(client: Client, storage_id: int, key: str) -> bool:
    try:
        return object_exists(client, storage_id, key)
    except ApiException as exc:
        if exc.status not in (403, 404):  # CVAT's answer when the bucket refuses the keys
            raise
        raise InvalidArgument(
            f"CVAT could not list the bucket to check whether {key} exists; check the bucket "
            "name" + keys_hint(storage_id)
        ) from exc


def _export(task, format_name: str, include_images: bool, storage_id: int, key: str) -> None:
    try:
        # The SDK starts the export and waits for it; CVAT uploads the file under the key.
        task.export_dataset(
            format_name,
            key,
            include_images=include_images,
            location=Location.CLOUD_STORAGE,
            cloud_storage_id=storage_id,
            status_check_period=1,  # the SDK default (5 s) makes even a tiny export take 5 s
        )
    except Exception as exc:
        if isinstance(exc, ApiException) and exc.status == 409:  # CVAT: one per task and format
            raise InvalidArgument(
                f"another export of task {task.id} as {format_name} is still running (CVAT runs "
                "one per task, format and include_images); wait for it, then call again"
            ) from exc
        if not is_refusal(exc):
            raise
        hint = f"check that {format_name} can export this task" + keys_hint(storage_id, "write to")
        raise InvalidArgument(f"{explain(exc).rstrip()}; {hint}") from exc


def export_dataset(
    client: Client,
    task_id: int,
    format_name: str,
    cloud_storage_id: int,
    cloud_path: str,
    include_images: bool,
    overwrite: bool,
) -> ExportResult:
    key = _export_key(cloud_path)
    task = client.tasks.retrieve(task_id)
    check_format(client, format_name, str(task.dimension))
    check_storage(client, cloud_storage_id, key)
    with _reserved(cloud_storage_id, key):
        exists = _exists(client, cloud_storage_id, key)
        if exists and not overwrite:
            raise InvalidArgument(
                f"cloud storage #{cloud_storage_id} already has {key}; ask the user whether to "
                "replace it, then call again with overwrite=true (or pick another cloud_path)"
            )
        started = time.monotonic()
        _export(task, format_name, include_images, cloud_storage_id, key)
    return ExportResult(
        cloud_storage_id=cloud_storage_id,
        cloud_path=key,
        replaced=exists,
        seconds=round(time.monotonic() - started, 1),
    )
