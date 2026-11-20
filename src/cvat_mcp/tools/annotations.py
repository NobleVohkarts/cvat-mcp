from mcp.server.mcpserver import Context, Image

from cvat_mcp import render
from cvat_mcp.server import get_client, tool
from cvat_mcp.services import annotations
from cvat_mcp.services.annotations import FrameAnnotations


@tool()
def get_frame_annotations(ctx: Context, task_id: int, frame: int) -> FrameAnnotations:
    """Use for exact object data on one frame: id, label, type, box [x1, y1, x2, y2] in original
    pixels, attributes and source. Use render_frame to see the frame; use these ids and boxes
    for create_issue."""
    return annotations.get_frame_annotations(get_client(ctx), task_id, frame)


@tool(structured_output=False)
def render_frame(ctx: Context, task_id: int, frame: int, max_side: int = 1280) -> list:
    """Use to look at a frame: returns the image (longest side up to max_side, 256..2048) with
    every object's box and `#id label` caption drawn on it, plus a text legend with exact boxes
    in original pixels. Check visually for missing objects, wrong labels and boxes that miss
    their object. Use get_frame_annotations for attributes."""
    jpeg, text = render.render_frame(get_client(ctx), task_id, frame, max_side)
    return [Image(data=jpeg, format="jpeg"), text]
