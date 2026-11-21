import json

from cvat_sdk import Client
from cvat_sdk.api_client import models
from pydantic import BaseModel

from cvat_mcp.errors import InvalidArgument


class Conflict(BaseModel):
    frame: int
    type: str
    shape_ids: list[int]  # annotation-job shapes, as get_frame_annotations shows them
    gt_shape_ids: list[int]  # shapes inside the GT job (not in the task's annotations)


class Score(BaseModel):
    name: str
    metric: str
    score: float | None
    threshold: float
    valid: int
    missing: int
    extra: int


class Report(BaseModel):
    report_id: int
    task_id: int
    gt_job_id: int
    url: str
    created_date: str
    total_frames: int
    validation_frames: int
    conflict_count: int
    error_count: int
    conflicts_by_type: dict[str, int]
    jobs: dict[str, int] | None
    scores: list[Score]
    conflicts: list[Conflict]
    conflicts_truncated: bool


def create_report(client: Client, base_url: str, task_id: int, max_conflicts: int) -> Report:
    """Compute a fresh quality report: the task's annotation jobs against its GT job."""
    task = client.tasks.retrieve(task_id)
    gt_job_id = next((j.id for j in task.get_jobs() if str(j.type) == "ground_truth"), None)
    if gt_job_id is None:
        raise InvalidArgument(
            f"task {task_id} has no ground-truth job: create one (create_gt_job) and put the "
            "reference annotations in it first"
        )

    quality_api = client.api_client.quality_api
    # CVAT answers 202 + rq_id, but the SDK's schema expects a QualityReport, so read it raw.
    _, response = quality_api.create_report(
        quality_report_create_request=models.QualityReportCreateRequest(task_id=task_id),
        _parse_response=False,
    )
    request, _ = client.wait_for_completion(
        json.loads(response.data)["rq_id"], status_check_period=1
    )
    report, _ = quality_api.retrieve_report(request.result_id)
    summary = report.summary

    conflicts, truncated = [], summary.conflict_count > 0
    if max_conflicts > 0:
        page, _ = quality_api.list_conflicts(
            report_id=report.id, page_size=max_conflicts, sort="frame", org_id=task.organization
        )
        conflicts = [_conflict(c, gt_job_id) for c in page.results]
        truncated = page.count > len(conflicts)

    return Report(
        report_id=report.id,
        task_id=task_id,
        gt_job_id=gt_job_id,
        url=f"{base_url.rstrip('/')}/tasks/{task_id}/quality-control",
        created_date=report.created_date.isoformat(),
        total_frames=summary.total_frames,
        validation_frames=summary.validation_frames,
        conflict_count=summary.conflict_count,
        error_count=summary.error_count,
        conflicts_by_type=dict(summary.conflicts_by_type),
        jobs=summary.jobs.to_dict() if getattr(summary, "jobs", None) else None,
        scores=[_score(item) for item in _requirement_items(summary)],
        conflicts=conflicts,
        conflicts_truncated=truncated,
    )


def _requirement_items(summary) -> list:
    requirements = getattr(summary, "requirements", None)  # optional in CVAT's summary
    return list(requirements.items) if requirements else []


def _score(item) -> Score:
    parts = item.score_components
    return Score(
        name=item.name,
        metric=str(item.metric),
        score=item.score,
        threshold=item.threshold,
        valid=parts.valid_count,
        missing=parts.missing_count,
        extra=parts.extra_count,
    )


def _conflict(conflict, gt_job_id: int) -> Conflict:
    ids = conflict.annotation_ids
    return Conflict(
        frame=conflict.frame,
        type=str(conflict.type),
        shape_ids=[a.obj_id for a in ids if a.job_id != gt_job_id],
        gt_shape_ids=[a.obj_id for a in ids if a.job_id == gt_job_id],
    )
