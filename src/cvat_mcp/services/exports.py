import time
from typing import Literal

from cvat_sdk import Client
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
    exists = object_exists(client, cloud_storage_id, key)
    if exists and not overwrite:
        raise InvalidArgument(
            f"cloud storage #{cloud_storage_id} already has {key}; ask the user whether to "
            "replace it, then call again with overwrite=true (or pick another cloud_path)"
        )

    started = time.monotonic()
    try:
        # The SDK starts the export and waits for it; CVAT uploads the file under the key.
        task.export_dataset(
            format_name,
            key,
            include_images=include_images,
            location=Location.CLOUD_STORAGE,
            cloud_storage_id=cloud_storage_id,
            status_check_period=1,  # the SDK default (5 s) makes even a tiny export take 5 s
        )
    except Exception as exc:
        if not is_refusal(exc):
            raise
        hint = f"check that {format_name} can export this task" + keys_hint(
            cloud_storage_id, "write to"
        )
        raise InvalidArgument(f"{explain(exc).rstrip()}; {hint}") from exc
    return ExportResult(
        cloud_storage_id=cloud_storage_id,
        cloud_path=key,
        replaced=exists,
        seconds=round(time.monotonic() - started, 1),
    )
