from typing import Annotated

from mcp.server.mcpserver import Context, Image
from pydantic import Field

from cvat_mcp import render
from cvat_mcp.server import get_client, settings, tool
from cvat_mcp.services import annotation_edits, annotations
from cvat_mcp.services.annotation_edits import AnnotationEdit, NewObject, ObjectChange
from cvat_mcp.services.annotations import FrameAnnotations

Ids = Annotated[list[int], Field(max_length=100)]


@tool()
def get_frame_annotations(ctx: Context, task_id: int, frame: int) -> FrameAnnotations:
    """Use for exact object data on one frame: id, label, type, box [x1, y1, x2, y2] in original
    pixels (plus points for non-rectangle shapes), attributes and source, and the frame's tags.
    Shape and tag ids are separate: a shape and a tag can share a number. Use render_frame to see
    the frame; use these ids and boxes for create_issue, update_annotations and
    delete_annotations."""
    return annotations.get_frame_annotations(get_client(ctx), task_id, frame)


@tool(structured_output=False)
def render_frame(
    ctx: Context, task_id: int, frame: int, max_side: int = 1280, include_clean: bool = False
) -> list:
    """Use to look at a frame: returns the image (longest side up to max_side, 256..2048) with
    every object's box and `#id label` caption drawn on it, plus a text legend with exact boxes
    in original pixels. Check visually for missing objects, wrong labels and boxes that miss
    their object. Set include_clean=True for crowded frames: you also get the same frame
    without annotations, so nothing is hidden under captions. Use get_frame_annotations for
    attributes."""
    jpegs, text = render.render_frame(get_client(ctx), task_id, frame, max_side, include_clean)
    return [*(Image(data=jpeg, format="jpeg") for jpeg in jpegs), text]


@tool(write=True)
def create_annotations(
    ctx: Context,
    task_id: int,
    objects: Annotated[list[NewObject], Field(min_length=1, max_length=100)],
) -> AnnotationEdit:
    """Use to add missing objects, only when the user asks to fix annotations (a review or
    filing issues is not such a request): rectangles by bbox [x1, y1, x2, y2], polygons,
    polylines and points by flat points, or frame tags, with a label of the task and optional
    attributes. Every object is checked first; then all are added in one step, or none. Returns
    the new ids and a CVAT UI link. Render the frame again afterwards to check."""
    return annotation_edits.create_annotations(get_client(ctx), settings.url, task_id, objects)


@tool(write=True)
def update_annotations(
    ctx: Context,
    task_id: int,
    objects: Annotated[list[ObjectChange], Field(min_length=1, max_length=100)],
) -> AnnotationEdit:
    """Use to correct existing objects, only when the user asks to fix annotations: change the
    label, the box (rectangles) or points (polygons, polylines, points), attributes (merged into
    the current values) or occluded, giving only what changes. Ids come from
    get_frame_annotations; set kind="tag" for a tag id. All changes apply in one step, or none.
    Tracks and ground-truth annotations are read-only here."""
    return annotation_edits.update_annotations(get_client(ctx), task_id, objects)


@tool(write=True, destructive_hint=True, idempotent_hint=True)
def delete_annotations(
    ctx: Context, task_id: int, shape_ids: Ids | None = None, tag_ids: Ids | None = None
) -> AnnotationEdit:
    """Use to remove wrong objects (e.g. a duplicate or a box around nothing), only when the user
    asks to fix annotations. Deletes exactly the given ids, all or none: shape ids and tag ids
    are separate (a shape and a tag can share a number), so pass each in its own list. Ids come
    from get_frame_annotations; unknown ids are refused and nothing is deleted."""
    return annotation_edits.delete_annotations(
        get_client(ctx), task_id, shape_ids or [], tag_ids or []
    )
