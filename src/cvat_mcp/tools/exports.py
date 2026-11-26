from mcp.server.mcpserver import Context

from cvat_mcp.server import get_client, tool
from cvat_mcp.services import exports
from cvat_mcp.services.exports import ExportResult


# destructive: overwrite=true replaces an object; open world: CVAT writes into the bucket
@tool(write=True, destructive_hint=True, open_world_hint=True)
def export_dataset(
    ctx: Context,
    task_id: int,
    format: str,  # pylint: disable=redefined-builtin
    cloud_storage_id: int,
    cloud_path: str,
    include_images: bool = False,
    overwrite: bool = False,
) -> ExportResult:
    """Use to save a task's annotations (optionally with images) as a dataset in a CVAT format
    such as 'COCO 1.0' or 'YOLO 1.1' into a CVAT cloud storage, only when the user asks.
    cloud_storage_id comes from list_cloud_storages; cloud_path is the object key to write,
    from the bucket root also for a storage with a prefix, e.g. exports/task-7-coco.zip (.zip
    is added if missing). An existing object is refused: ask the user whether to replace it,
    then call again with overwrite=true. Waits for CVAT to prepare the file and upload it to
    the bucket; returns the key, not the contents. Big exports with images can take
    minutes."""
    return exports.export_dataset(
        get_client(ctx), task_id, format, cloud_storage_id, cloud_path, include_images, overwrite
    )
