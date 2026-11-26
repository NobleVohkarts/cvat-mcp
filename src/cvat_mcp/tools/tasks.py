from typing import Annotated

from mcp.server.mcpserver import Context
from pydantic import Field

from cvat_mcp.server import get_client, settings, tool
from cvat_mcp.services import task_edits, tasks
from cvat_mcp.services.task_edits import MAX_FILES, CreatedTask, DeletedTask, LabelSpec, UpdatedTask
from cvat_mcp.services.tasks import TaskList, TaskOverview

Paths = Annotated[list[str], Field(min_length=1, max_length=MAX_FILES)]


@tool()
def list_tasks(
    ctx: Context, project_id: int | None = None, search: str | None = None, limit: int = 20
) -> TaskList:
    """Use to find tasks: returns up to `limit` tasks (newest first, max 100) with frame count,
    status and job progress, plus `total`, the number of all matching tasks. `search` is CVAT's
    free-text search: it matches task names but also project names, owners and assignees.
    Use get_task_overview for one task's labels, jobs and object counts."""
    return tasks.list_tasks(get_client(ctx), project_id, search, limit)


@tool()
def get_task_overview(ctx: Context, task_id: int) -> TaskOverview:
    """Use after list_tasks to understand one task: labels, jobs with their frame ranges, stage,
    state and assignee, and object counts per label. Needed before reviewing frames or creating
    issues: an issue goes to the job of type "annotation" whose frame range contains the frame
    (ground-truth and consensus jobs can overlap those ranges)."""
    return tasks.get_task_overview(get_client(ctx), task_id)


# open world: with urls or a cloud storage, CVAT downloads from any host
@tool(write=True, open_world_hint=True)
def create_task(
    ctx: Context,
    name: str,
    labels: list[LabelSpec] | None = None,
    project_id: int | None = None,
    files: Paths | None = None,
    urls: Paths | None = None,
    cloud_storage_id: int | None = None,
    cloud_path: str | None = None,
    segment_size: int | None = None,
) -> CreatedTask:
    """Use to create a task, only when the user asks: images from files or folders inside the
    server's work directory (folders expand recursively, hidden files skipped; all images are
    sorted by name in natural order, file names must be unique), from http(s) URLs, or from a
    CVAT cloud storage (cloud_storage_id from list_cloud_storages or create_cloud_storage, with
    cloud_path: an object key, a folder or a glob in that bucket; a folder is taken
    recursively, a path without a file extension counts as a folder (end a dotted folder name
    like v1.2 with /), * and ? are globs and * also matches across folders; an empty cloud_path
    lists the whole bucket, which is slow on big buckets). A bucket that has no cloud storage
    yet is registered first with create_cloud_storage, only when the user asks. Frames are
    named by their full key. Use cloud_storage_id + cloud_path, not urls, for bucket objects.
    Labels are given inline or taken from a project (project_id). Waits until CVAT has
    processed the data; if it fails, no task is left behind. Returns the task id, frame count,
    jobs, the cloud storage id and a CVAT UI link."""
    return task_edits.create_task(
        get_client(ctx),
        settings.url,
        settings.mcp_workdir,
        name,
        labels,
        project_id,
        files,
        urls,
        cloud_storage_id,
        cloud_path,
        segment_size,
    )


@tool(write=True)
def update_task(
    ctx: Context,
    task_id: int,
    name: str | None = None,
    assignee: str | None = None,
    project_id: int | None = None,
    add_labels: list[LabelSpec] | None = None,
) -> UpdatedTask:
    """Use to rename a task, assign it (a CVAT username; "" unassigns), move it into a project
    or add labels, only when the user asks. Give only what changes; a project move goes alone in
    its own call. Labels can be added to tasks outside a project, never removed here (removing a
    label deletes its annotations)."""
    return task_edits.update_task(get_client(ctx), task_id, name, assignee, project_id, add_labels)


@tool(write=True, destructive_hint=True, idempotent_hint=True)
def delete_task(ctx: Context, task_id: int) -> DeletedTask:
    """Use to delete a task, only when the user explicitly asks: removes it with all its jobs,
    annotations and issues, which can't be undone. Only tasks owned by this server's CVAT account
    can be deleted; others are refused and must be handled in the CVAT UI."""
    return task_edits.delete_task(get_client(ctx), task_id)
