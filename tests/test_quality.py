from datetime import datetime
from unittest.mock import Mock

import pytest
from cvat_sdk.api_client.exceptions import ApiException
from fakes import Obj, fake_client, fake_list_settings, fake_task
from fakes import quality_requirement as req

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.quality import create_report

JOBS = (Obj(id=3, type="annotation"), Obj(id=4, type="ground_truth"))


def _ann(obj_id, job_id, kind="shape"):
    return Obj(obj_id=obj_id, job_id=job_id, type=kind)


def _conflict(frame, type_, *ids):
    return Obj(frame=frame, type=type_, annotation_ids=list(ids))


def _summary(conflict_count=2, score=0.5, enabled=1):
    item = Obj(
        name="default",
        metric="accuracy",
        score=score,
        threshold=0.8,
        score_components=Obj(valid_count=4, missing_count=3, extra_count=1),
        calculation=Obj(status="completed", reason=None),
    )
    return Obj(
        total_frames=6,
        validation_frames=3,
        conflict_count=conflict_count,
        error_count=conflict_count,
        conflicts_by_type={"missing_annotation": 1, "mismatching_label": 1},
        jobs=Obj(to_dict=lambda: {"total": 2, "not_checkable": 0}),
        requirements=Obj(enabled=enabled, items=[item] if enabled else []),
    )


def _client(conflicts=(), jobs=JOBS, summary=None, organization=None, count=None):
    report = Obj(id=9, created_date=datetime(2026, 11, 21), summary=summary or _summary())
    quality_api = Obj(
        create_report=Mock(return_value=(None, Obj(data=b'{"rq_id": "rq-1"}'))),
        retrieve_report=Mock(return_value=(report, None)),
        list_conflicts=Mock(
            return_value=(
                Obj(count=len(conflicts) if count is None else count, results=list(conflicts)),
                None,
            )
        ),
        list_settings=fake_list_settings(
            1, [], [req(1, "Base tag", "tag"), req(2, "Base rectangle", "rectangle")]
        ),
    )
    return fake_client(
        fake_task(jobs=jobs, organization=organization, project_id=9),
        api_client=Obj(quality_api=quality_api),
        wait_for_completion=Mock(return_value=(Obj(result_id=9), None)),
    )


def test_create_report_waits_and_summarises():
    client = _client()
    result = create_report(client, "http://cvat/", 1, 100)

    api = client.api_client.quality_api
    _, kwargs = api.create_report.call_args
    assert kwargs["quality_report_create_request"].task_id == 1
    assert kwargs["_parse_response"] is False
    client.wait_for_completion.assert_called_once_with("rq-1", status_check_period=1)
    api.retrieve_report.assert_called_once_with(9)

    assert result.report_id == 9
    assert result.task_id == 1
    assert result.gt_job_id == 4
    assert result.url == "http://cvat/tasks/1/quality-control"
    assert result.created_date.startswith("2026-11-21")
    assert (result.total_frames, result.validation_frames) == (6, 3)
    assert (result.conflict_count, result.error_count) == (2, 2)
    assert result.conflicts_by_type == {"missing_annotation": 1, "mismatching_label": 1}
    assert result.jobs == {"total": 2, "not_checkable": 0}
    [score] = result.scores
    assert (score.name, score.metric, score.score, score.threshold) == (
        "default",
        "accuracy",
        0.5,
        0.8,
    )
    assert (score.valid, score.missing, score.extra) == (4, 3, 1)
    assert (score.status, score.reason) == ("completed", None)


def _refs(refs):
    return [(r.id, r.kind) for r in refs]


def test_conflict_ids_split_by_job():
    conflicts = [
        _conflict(1, "mismatching_label", _ann(11, 3), _ann(21, 4)),
        _conflict(3, "missing_annotation", _ann(21, 4)),
        _conflict(4, "extra_annotation", _ann(11, 3, "tag")),
        _conflict(5, "mismatching_label", _ann(12, 3, "track"), _ann(22, 4)),
    ]
    client = _client(conflicts)
    result = create_report(client, "http://cvat", 1, 100)

    _, kwargs = client.api_client.quality_api.list_conflicts.call_args
    assert (kwargs["report_id"], kwargs["page_size"], kwargs["sort"]) == (9, 100, "frame")
    first, second, tag, track = result.conflicts
    assert (first.frame, first.type) == (1, "mismatching_label")
    assert (_refs(first.ids), _refs(first.gt_ids)) == ([(11, "shape")], [(21, "shape")])
    assert (_refs(second.ids), _refs(second.gt_ids)) == ([], [(21, "shape")])
    # shape, tag and track ids are separate sequences in CVAT: the kind travels with the id
    assert _refs(tag.ids) == [(11, "tag")]
    assert (_refs(track.ids), _refs(track.gt_ids)) == ([(12, "track")], [(22, "shape")])


@pytest.mark.parametrize("count, truncated", [(5, True), (2, False)])
def test_conflicts_truncated(count, truncated):
    # truncation comes from the listing: with overlapping jobs it holds more rows than the
    # summary's deduplicated conflict_count
    conflicts = [_conflict(f, "missing_annotation", _ann(21, 4)) for f in (1, 3)]
    client = _client(conflicts, summary=_summary(conflict_count=2), count=count)
    result = create_report(client, "http://cvat", 1, 2)
    assert len(result.conflicts) == 2
    assert result.conflicts_truncated is truncated


@pytest.mark.parametrize("max_conflicts, conflict_count", [(0, 0), (0, 2), (100, 0)])
def test_listing_skipped_without_need(max_conflicts, conflict_count):
    client = _client(summary=_summary(conflict_count=conflict_count))
    result = create_report(client, "http://cvat", 1, max_conflicts)
    client.api_client.quality_api.list_conflicts.assert_not_called()
    assert result.conflicts == []
    assert result.conflicts_truncated is (conflict_count > 0)


def test_null_score_passes_through():
    result = create_report(_client(summary=_summary(score=None)), "http://cvat", 1, 100)
    assert result.scores[0].score is None


def test_no_gt_job_refused():
    client = _client(jobs=[Obj(id=3, type="annotation")])
    with pytest.raises(InvalidArgument, match="no ground-truth job.*create_gt_job"):
        create_report(client, "http://cvat", 1, 100)
    client.api_client.quality_api.create_report.assert_not_called()


@pytest.mark.parametrize("organization, org_args", [(None, {"org": ""}), (7, {"org_id": 7})])
def test_conflicts_org_context(organization, org_args):
    # the raw SDK endpoint refuses org_id=None; like the SDK's list helpers, send org="" instead
    client = _client([_conflict(1, "missing_annotation", _ann(21, 4))], organization=organization)
    create_report(client, "http://cvat", 1, 100)
    _, kwargs = client.api_client.quality_api.list_conflicts.call_args
    assert {k: kwargs[k] for k in ("org", "org_id") if k in kwargs} == org_args


def test_no_enabled_requirement_refused():
    # CVAT starts every quality requirement disabled; then a report compares nothing
    client = _client(summary=_summary(conflict_count=0, enabled=0))
    with pytest.raises(InvalidArgument, match="no quality requirement is enabled") as refused:
        create_report(client, "http://cvat", 1, 100)
    message = str(refused.value)
    # the state the agent needs to decide: where the settings live and what exists
    assert "uses project #9's settings" in message
    assert "Base rectangle (rectangle)" in message
    assert "update_quality_settings" in message
    client.api_client.quality_api.list_conflicts.assert_not_called()


def test_overlap_duplicates_dropped():
    # overlapping jobs report the same conflict once per job
    same = [_conflict(2, "missing_annotation", _ann(21, 4)) for _ in range(2)]
    result = create_report(_client(same), "http://cvat", 1, 100)
    assert len(result.conflicts) == 1


def test_running_report_is_awaited():
    # CVAT answers 409 + the running request's rq_id when this task's report is being computed
    busy = ApiException(status=409, reason="Conflict")
    busy.body = b'{"rq_id": "rq-0"}'
    client = _client()
    client.api_client.quality_api.create_report.side_effect = busy
    assert create_report(client, "http://cvat", 1, 100).report_id == 9
    client.wait_for_completion.assert_called_once_with("rq-0", status_check_period=1)


def test_other_api_errors_propagate():
    client = _client()
    client.api_client.quality_api.create_report.side_effect = ApiException(status=400)
    with pytest.raises(ApiException):
        create_report(client, "http://cvat", 1, 100)


def test_not_computed_score_shows_status():
    summary = _summary(score=None)
    summary.requirements.items[0].calculation = Obj(status="not_computed", reason="no_annotations")
    [score] = create_report(_client(summary=summary), "http://cvat", 1, 100).scores
    assert (score.score, score.status, score.reason) == (None, "not_computed", "no_annotations")
