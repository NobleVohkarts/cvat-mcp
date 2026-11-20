from mcp.server.mcpserver import Context

from cvat_mcp.server import get_client, settings, tool
from cvat_mcp.services import exports
from cvat_mcp.services.exports import ExportResult


# Reads CVAT only, so it exists in both modes; it does create a local file.
@tool(read_only_hint=False, destructive_hint=False, idempotent_hint=False)
def export_dataset(
    ctx: Context,
    task_id: int,
    format: str,  # pylint: disable=redefined-builtin
    include_images: bool = False,
    filename: str | None = None,
) -> ExportResult:
    """Use to save a task's annotations (optionally with images) as a dataset file in a CVAT
    format such as 'COCO 1.0' or 'YOLO 1.1'. Waits for CVAT to prepare it and saves it inside
    the server's work directory; never overwrites (adds -1, -2, ...). Returns the file path and
    size, not the contents."""
    return exports.export_dataset(
        get_client(ctx), settings.mcp_workdir, task_id, format, include_images, filename
    )
