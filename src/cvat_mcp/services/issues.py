from cvat_sdk import Client
from cvat_sdk.api_client import models
from pydantic import BaseModel

from cvat_mcp.errors import InvalidArgument, explain
from cvat_mcp.services.annotations import check_frame, shape_bbox
from cvat_mcp.services.tasks import job_url
from cvat_mcp.services.users import is_own, require_own


class IssueRef(BaseModel):
    issue_id: int
    job_id: int
    frame: int
    position: list[float]
    url: str


class Reply(BaseModel):
    author: str | None
    message: str


class IssueInfo(BaseModel):
    issue_id: int
    job_id: int
    frame: int
    position: list[float]
    message: str
    resolved: bool
    author: str | None
    replies: list[Reply]
    own: bool


class IssueList(BaseModel):
    task_id: int
    frame: int | None
    issues: list[IssueInfo]


class UpdatedIssue(BaseModel):
    issue_id: int
    job_id: int
    frame: int
    resolved: bool
    reply_id: int | None  # the comment this call posted
    url: str


class DeletedIssue(BaseModel):
    issue_id: int
    job_id: int
    frame: int


def check_bbox(bbox: list[float], name: str = "bbox") -> None:
    if not (len(bbox) == 4 and bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        raise InvalidArgument(
            f"{name} must be [x1, y1, x2, y2] in original pixels with x1 < x2 and y1 < y2"
        )


def find_job_for_frame(jobs, frame: int):
    """The annotation job whose frame range contains the frame (issues belong there)."""
    return next(
        (j for j in jobs if str(j.type) == "annotation" and j.start_frame <= frame <= j.stop_frame),
        None,
    )


def create_issue(
    client: Client,
    base_url: str,
    job_id: int,
    frame: int,
    message: str,
    object_id: int | None,
    bbox: list[float] | None,
) -> IssueRef:
    # Argument checks first: they need no requests.
    if not message.strip():
        raise InvalidArgument("message must say what is wrong and what to do")
    if (object_id is None) == (bbox is None):
        raise InvalidArgument("pass exactly one of object_id or bbox")
    if bbox is not None:
        check_bbox(bbox)

    job = client.jobs.retrieve(job_id)
    if str(job.type) != "annotation" or not job.start_frame <= frame <= job.stop_frame:
        right = find_job_for_frame(client.tasks.retrieve(job.task_id).get_jobs(), frame)
        hint = (
            f"use job {right.id} (frames {right.start_frame}..{right.stop_frame})"
            if right
            else "no annotation job contains this frame; check get_task_overview"
        )
        raise InvalidArgument(
            f"frame {frame} is not in annotation job {job_id} "
            f"({job.type}, frames {job.start_frame}..{job.stop_frame}); {hint}"
        )

    if object_id is not None:
        shapes = {s.id: s for s in job.get_annotations().shapes if s.frame == frame}
        if object_id not in shapes:
            raise InvalidArgument(
                f"object #{object_id} is not on frame {frame}; "
                "use get_frame_annotations to see the ids on this frame"
            )
        position = shape_bbox(shapes[object_id])
    else:
        position = list(bbox)

    # CVAT expects the task frame number here and does not check it against the job's range.
    issue = client.issues.create(
        models.IssueWriteRequest(frame=frame, position=position, job=job_id, message=message)
    )
    return IssueRef(
        issue_id=issue.id,
        job_id=job_id,
        frame=frame,
        position=position,
        url=job_url(base_url, job.task_id, job_id, frame),
    )


def list_issues(client: Client, task_id: int, frame: int | None) -> IssueList:
    task = client.tasks.retrieve(task_id)
    filters = {}
    if frame is not None:
        check_frame(frame, task.size)
        filters["frame_id"] = frame
    # An explicit org context: without it the comments endpoint answers an empty list.
    found = client.issues.list(org_id=task.organization, task_id=task_id, **filters)
    # CVAT keeps an issue's message as its first comment; comments filter by job, not task.
    messages, replies = {}, {}
    for job_id in {i.job for i in found}:
        comments = client.comments.list(org_id=task.organization, job_id=job_id, **filters)
        for c in sorted(comments, key=lambda c: c.id):  # the first comment is the message
            if c.issue in messages:
                author = c.owner.username if c.owner else None
                replies.setdefault(c.issue, []).append(Reply(author=author, message=c.message))
            else:
                messages[c.issue] = c.message
    me = client.users.retrieve_current_user().id
    return IssueList(
        task_id=task_id,
        frame=frame,
        issues=[
            IssueInfo(
                issue_id=i.id,
                job_id=i.job,
                frame=i.frame,
                position=i.position,
                message=messages.get(i.id, ""),
                resolved=i.resolved,
                author=i.owner.username if i.owner else None,
                replies=replies.get(i.id, []),
                own=is_own(i, me),
            )
            for i in sorted(found, key=lambda i: (i.frame, i.id))
        ],
    )


def update_issue(
    client: Client, base_url: str, issue_id: int, reply: str | None, resolved: bool | None
) -> UpdatedIssue:
    """Reply to an issue and/or resolve or reopen it; any issue CVAT lets this user change."""
    if reply is None and resolved is None:
        raise InvalidArgument("pass reply, resolved or both")
    if reply is not None and not reply.strip():
        raise InvalidArgument("reply must say something; leave it out to only change resolved")

    issue = client.issues.retrieve(issue_id)
    job = client.jobs.retrieve(issue.job)  # before the reply: nothing may fail after it unsaid
    comment = None
    if reply is not None:  # first, so a reply-and-close keeps its reason even if the patch fails
        # CVAT checks a create in the request's org context, not the issue's.
        org = {"org_id": job.organization} if job.organization is not None else {}
        comment, _ = client.api_client.comments_api.create(
            models.CommentWriteRequest(issue=issue_id, message=reply), **org
        )
    if resolved is not None and resolved != issue.resolved:
        try:  # one PATCH: the SDK proxy's update() would GET the issue again
            patch = models.PatchedIssueWriteRequest(resolved=resolved)
            client.api_client.issues_api.partial_update(issue_id, patched_issue_write_request=patch)
        except Exception as error:
            if comment is None:
                raise
            # Said explicitly, so the model doesn't post the same reply again on a retry.
            raise InvalidArgument(
                f"the reply was posted (comment #{comment.id}), but changing resolved failed: "
                f"{explain(error) or error}"
            ) from error
    return UpdatedIssue(
        issue_id=issue_id,
        job_id=issue.job,
        frame=issue.frame,
        resolved=issue.resolved if resolved is None else resolved,
        reply_id=comment.id if comment else None,
        url=job_url(base_url, job.task_id, issue.job, issue.frame),
    )


def delete_issue(client: Client, issue_id: int) -> DeletedIssue:
    """Delete one issue, only if this server's user filed it."""
    issue = client.issues.retrieve(issue_id)
    require_own(client, issue, f"issue #{issue_id}")
    issue.remove()
    return DeletedIssue(issue_id=issue_id, job_id=issue.job, frame=issue.frame)
