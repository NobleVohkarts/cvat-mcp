from typing import Annotated

from mcp.server.mcpserver import Context, Image
from mcp.types import EmbeddedResource, TextResourceContents
from pydantic import Field

from cvat_mcp import render
from cvat_mcp.server import get_client, settings, tool
from cvat_mcp.services import annotation_edits, annotation_imports, annotations
from cvat_mcp.services.annotation_edits import AnnotationEdit, NewObject, ObjectChange
from cvat_mcp.services.annotation_imports import DEFAULT_FORMAT, ImportedAnnotations, Mode
from cvat_mcp.services.annotations import FrameAnnotations

Ids = Annotated[list[int], Field(max_length=100)]
JobId = Annotated[
    int | None,
    Field(
        description="a job of the task: use its annotations instead of the task's; ground-truth "
        "jobs (see get_task_overview) are only reachable this way, and frames must be the job's. "
        "Edit ground truth only when the user explicitly asks: annotators are measured against it"
    ),
]


@tool()
def get_frame_annotations(
    ctx: Context, task_id: int, frame: int, job_id: JobId = None
) -> FrameAnnotations:
    """Use for exact object data on one frame: id, label, type, box [x1, y1, x2, y2] in original
    pixels (plus points for polygons, polylines and points), attributes and source, and the frame's tags.
    Shape and tag ids are separate: a shape and a tag can share a number. Use render_frame to see
    the frame; use these ids and boxes for create_issue, update_annotations and
    delete_annotations."""
    return annotations.get_frame_annotations(get_client(ctx), task_id, frame, job_id)


@tool(structured_output=False)
def download_annotations(
    ctx: Context,
    task_id: int | None = None,
    job_id: Annotated[
        int | None,
        Field(description="one job (any type, ground truth included) instead of the task's jobs"),
    ] = None,
) -> list[EmbeddedResource]:
    """Use to read all annotations of a job at once (job_id: any job, ground truth included) or
    of every annotation job of a task (task_id): one compact JSON per job with its frames (name,
    size), shapes (id, frame, label, type, points as flat x, y in original pixels, rotation in
    degrees when not 0, group, occluded, z_order, source, attributes) and tags. Points are
    CVAT's own: masks hold RLE counts followed by [left, top, right, bottom], and a skeleton's
    points are empty (its keypoints are not included). Tracks are not included (counted in
    skipped_tracks). Refuses above 2000 shapes in total; then pass job_id or use
    get_frame_annotations frame by frame. Pass exactly one of task_id or job_id."""
    return [
        EmbeddedResource(
            type="resource",
            resource=TextResourceContents(
                uri=f"cvat://jobs/{job.job_id}/annotations.json",
                mimeType="application/json",
                text=job.model_dump_json(exclude_none=True),
            ),
        )
        for job in annotations.download_annotations(get_client(ctx), task_id, job_id)
    ]


@tool(structured_output=False)
def render_frame(
    ctx: Context,
    task_id: int,
    frame: int,
    max_side: int = 1280,
    include_clean: bool = False,
    job_id: JobId = None,
) -> list:
    """Use to look at a frame: returns the image (longest side up to max_side, 256..2048) with
    every object drawn on it (rectangles as boxes, polygons and polylines by their points,
    points as dots) with an `#id label` caption, plus a text legend with exact boxes in original
    pixels. Check visually for missing objects, wrong labels and boxes that miss
    their object. Set include_clean=True for crowded frames: you also get the same frame
    without annotations, so nothing is hidden under captions. Use get_frame_annotations for
    attributes."""
    jpegs, text = render.render_frame(
        get_client(ctx), task_id, frame, max_side, include_clean, job_id
    )
    return [*(Image(data=jpeg, format="jpeg") for jpeg in jpegs), text]


@tool(write=True)
def create_annotations(
    ctx: Context,
    task_id: int,
    objects: Annotated[list[NewObject], Field(min_length=1, max_length=100)],
    job_id: JobId = None,
) -> AnnotationEdit:
    """Use to add missing objects, only when the user asks to fix annotations (a review or
    filing issues is not such a request): rectangles by bbox [x1, y1, x2, y2], polygons,
    polylines and points by flat points, or frame tags, with a label of the task and optional
    attributes. Every object is checked first; then all are added in one step, or none. Returns
    the new ids and a CVAT UI link (an id of null means the object was added but its id could not
    be matched: do not retry, use get_frame_annotations). Render the frame again to check."""
    return annotation_edits.create_annotations(
        get_client(ctx), settings.url, task_id, objects, job_id
    )


@tool(write=True)
def update_annotations(
    ctx: Context,
    task_id: int,
    objects: Annotated[list[ObjectChange], Field(min_length=1, max_length=100)],
    job_id: JobId = None,
) -> AnnotationEdit:
    """Use to correct existing objects, only when the user asks to fix annotations: change the
    label, the box (rectangles) or points (polygons, polylines, points), attributes (merged into
    the current values) or occluded, giving only what changes. Ids come from
    get_frame_annotations; set kind="tag" for a tag id. All changes apply in one step, or none.
    Tracks are read-only here."""
    return annotation_edits.update_annotations(get_client(ctx), task_id, objects, job_id)


@tool(write=True, destructive_hint=True, idempotent_hint=True)
def delete_annotations(
    ctx: Context,
    task_id: int,
    shape_ids: Ids | None = None,
    tag_ids: Ids | None = None,
    job_id: JobId = None,
) -> AnnotationEdit:
    """Use to remove wrong objects (e.g. a duplicate or a box around nothing), only when the user
    asks to fix annotations. Deletes exactly the given ids, all or none: shape ids and tag ids
    are separate (a shape and a tag can share a number), so pass each in its own list. Ids come
    from get_frame_annotations; unknown ids are refused and nothing is deleted."""
    return annotation_edits.delete_annotations(
        get_client(ctx), task_id, shape_ids or [], tag_ids or [], job_id
    )


# open world: CVAT reads a cloud file from the bucket's host; destructive: mode="replace"
@tool(write=True, destructive_hint=True, open_world_hint=True)
def import_annotations(
    ctx: Context,
    job_id: int | None = None,
    task_id: int | None = None,
    cloud_storage_id: int | None = None,
    cloud_path: str | None = None,
    file: str | None = None,
    format: str = DEFAULT_FORMAT,  # pylint: disable=redefined-builtin  # CVAT's name for it
    mode: Mode = "append",
) -> ImportedAnnotations:
    """Use to load an annotation file into CVAT, only when the user asks: into a job (job_id: an
    annotation or ground-truth job) or a whole task (task_id). The file is either in a CVAT
    cloud storage (cloud_storage_id from list_cloud_storages or create_cloud_storage, with
    cloud_path: the object key of one file, e.g. annotations/labels.json; CVAT reads it through
    the storage), or file, a path in the server's work directory. format is
    any CVAT import format (default COCO 1.0). The image file names in the file must match the
    task's frame names (a missing folder prefix is fine); a ground-truth job keeps only its own
    frames, so a file for all images works. mode "append" adds to the annotations already there
    (importing the same file twice duplicates the objects); "replace" first deletes all the
    target's annotations, so use it only when the user asks to replace them. Returns the object
    counts after the import. A file CVAT can't import takes about a minute to fail (CVAT retries
    it); wait for the answer instead of calling again."""
    return annotation_imports.import_annotations(
        get_client(ctx),
        settings.mcp_workdir,
        job_id,
        task_id,
        cloud_storage_id,
        cloud_path,
        file,
        format,
        mode,
    )
