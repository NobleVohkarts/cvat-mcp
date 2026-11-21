from cvat_sdk import Client
from cvat_sdk.api_client import models
from pydantic import BaseModel

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.annotations import check_frame, shape_bbox


class IssueRef(BaseModel):
    issue_id: int
    job_id: int
    frame: int
    position: list[float]
    url: str


class IssueInfo(BaseModel):
    issue_id: int
    job_id: int
    frame: int
    position: list[float]
    message: str
    resolved: bool
    author: str | None
    replies: int
    own: bool


class IssueList(BaseModel):
    task_id: int
    frame: int | None
    issues: list[IssueInfo]


class DeletedIssue(BaseModel):
    issue_id: int
    job_id: int
    frame: int


def check_bbox(bbox: list[float]) -> None:
    if not (len(bbox) == 4 and bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        raise InvalidArgument(
            "bbox must be [x1, y1, x2, y2] in original pixels with x1 < x2 and y1 < y2"
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
        url=f"{base_url.rstrip('/')}/tasks/{job.task_id}/jobs/{job_id}?frame={frame}",
    )


def _is_own(issue, user_id: int) -> bool:
    """Filed by this server's CVAT user: the only issues delete_issue may remove."""
    return issue.owner is not None and issue.owner.id == user_id


def list_issues(client: Client, task_id: int, frame: int | None) -> IssueList:
    task = client.tasks.retrieve(task_id)
    filters = {}
    if frame is not None:
        check_frame(frame, task.size)
        filters["frame_id"] = frame
    # An explicit org context: without it the comments endpoint answers an empty list.
    found = client.issues.list(org_id=task.organization, task_id=task_id, **filters)
    # CVAT keeps an issue's message as its first comment; comments filter by job, not task.
    messages = {}
    for job_id in {i.job for i in found}:
        comments = client.comments.list(org_id=task.organization, job_id=job_id, **filters)
        for comment in sorted(comments, key=lambda c: c.id):
            messages.setdefault(comment.issue, comment.message)
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
                replies=max(i.comments.count - 1, 0),  # the first comment is the message
                own=_is_own(i, me),
            )
            for i in sorted(found, key=lambda i: (i.frame, i.id))
        ],
    )


def delete_issue(client: Client, issue_id: int) -> DeletedIssue:
    """Delete one issue, only if this server's user filed it."""
    issue = client.issues.retrieve(issue_id)
    me = client.users.retrieve_current_user()
    if not _is_own(issue, me.id):
        author = issue.owner.username if issue.owner else "a deleted user"
        raise InvalidArgument(
            f"issue #{issue_id} was filed by {author}, not by this server's user {me.username}; "
            "only its own issues can be deleted here, others in the CVAT UI"
        )
    issue.remove()
    return DeletedIssue(issue_id=issue_id, job_id=issue.job, frame=issue.frame)
