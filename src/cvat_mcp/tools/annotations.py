from mcp.server.mcpserver import Context

from cvat_mcp.server import get_client, tool
from cvat_mcp.services import annotations
from cvat_mcp.services.annotations import FrameAnnotations


@tool()
def get_frame_annotations(ctx: Context, task_id: int, frame: int) -> FrameAnnotations:
    """Use for exact object data on one frame: id, label, type, box [x1, y1, x2, y2] in original
    pixels, attributes and source. Use render_frame to see the frame; use these ids and boxes
    for create_issue."""
    return annotations.get_frame_annotations(get_client(ctx), task_id, frame)
