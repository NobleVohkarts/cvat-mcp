import json

from cvat_sdk import Client
from cvat_sdk.api_client import models
from cvat_sdk.api_client.exceptions import ApiException
from pydantic import BaseModel

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.issues import job_url


class Ref(BaseModel):
    id: int
    kind: str  # shape, tag or track: CVAT numbers each kind separately


class Conflict(BaseModel):
    frame: int
    type: str
    ids: list[Ref]  # annotation-job objects, as get_frame_annotations shows shapes and tags
    gt_ids: list[Ref]  # objects inside the GT job (not in the task's annotations)


class Score(BaseModel):
    name: str
    metric: str
    score: float | None
    threshold: float
    valid: int
    missing: int
    extra: int
    status: str  # CVAT's calculation status; score null + "not_computed" = nothing compared
    reason: str | None


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
    request, _ = client.wait_for_completion(
        _start_report(quality_api, task_id), status_check_period=1
    )
    report, _ = quality_api.retrieve_report(request.result_id)
    summary = report.summary
    requirements = getattr(summary, "requirements", None)  # optional in CVAT's summary
    if requirements is not None and requirements.enabled == 0:
        # CVAT starts with every requirement disabled; such a report compares nothing
        raise InvalidArgument(
            f"no quality requirement is enabled for task {task_id}, so CVAT compared nothing: "
            "in the CVAT UI open the task's Quality control > Settings and enable the "
            "requirement for the GT shapes' type (e.g. 'Base rectangle' for boxes; in the "
            "project's settings if the task inherits them), then call create_report again"
        )

    conflicts, truncated = [], summary.conflict_count > 0
    if max_conflicts > 0 and summary.conflict_count > 0:
        # The SDK's list helpers send org="" (personal workspace) for org_id=None; the raw
        # endpoint refuses None, so do the same here.
        org = {"org_id": task.organization} if task.organization is not None else {"org": ""}
        page, _ = quality_api.list_conflicts(
            report_id=report.id, page_size=max_conflicts, sort="frame", **org
        )
        found = [_conflict(c, gt_job_id) for c in page.results]
        # overlapping jobs can report the same conflict once per job
        conflicts = list({c.model_dump_json(): c for c in found}.values())
        truncated = page.count > len(page.results)

    return Report(
        report_id=report.id,
        task_id=task_id,
        gt_job_id=gt_job_id,
        url=f"{job_url(base_url, task_id)}/quality-control",
        created_date=report.created_date.isoformat(),
        total_frames=summary.total_frames,
        validation_frames=summary.validation_frames,
        conflict_count=summary.conflict_count,
        error_count=summary.error_count,
        conflicts_by_type=dict(summary.conflicts_by_type),
        jobs=summary.jobs.to_dict() if getattr(summary, "jobs", None) else None,
        scores=[_score(item) for item in (requirements.items if requirements else [])],
        conflicts=conflicts,
        conflicts_truncated=truncated,
    )


def _start_report(quality_api, task_id: int) -> str:
    """Start the task's report; the rq_id of it, or of the one already being computed."""
    try:
        # CVAT answers 202 + rq_id, but the SDK's schema expects a QualityReport: read it raw.
        _, response = quality_api.create_report(
            quality_report_create_request=models.QualityReportCreateRequest(task_id=task_id),
            _parse_response=False,
        )
        body = response.data
    except ApiException as exc:
        if exc.status != 409:  # 409: this task's report is already being computed
            raise
        body = exc.body
    return json.loads(body)["rq_id"]


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
        status=str(item.calculation.status),
        reason=str(item.calculation.reason) if getattr(item.calculation, "reason", None) else None,
    )


def _conflict(conflict, gt_job_id: int) -> Conflict:
    def refs(in_gt: bool) -> list[Ref]:
        return [
            Ref(id=a.obj_id, kind=str(a.type))
            for a in conflict.annotation_ids
            if (a.job_id == gt_job_id) is in_gt
        ]

    return Conflict(
        frame=conflict.frame, type=str(conflict.type), ids=refs(False), gt_ids=refs(True)
    )
