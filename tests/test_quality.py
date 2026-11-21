from datetime import datetime
from unittest.mock import Mock

import pytest
from fakes import Obj, fake_client, fake_task

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.quality import create_report

JOBS = (Obj(id=3, type="annotation"), Obj(id=4, type="ground_truth"))


def _ann(obj_id, job_id):
    return Obj(obj_id=obj_id, job_id=job_id)


def _conflict(frame, type_, *ids):
    return Obj(frame=frame, type=type_, annotation_ids=list(ids))


def _summary(conflict_count=2, score=0.5):
    item = Obj(
        name="default",
        metric="accuracy",
        score=score,
        threshold=0.8,
        score_components=Obj(valid_count=4, missing_count=3, extra_count=1),
    )
    return Obj(
        total_frames=6,
        validation_frames=3,
        conflict_count=conflict_count,
        error_count=conflict_count,
        conflicts_by_type={"missing_annotation": 1, "mismatching_label": 1},
        jobs=Obj(to_dict=lambda: {"total": 2, "not_checkable": 0}),
        requirements=Obj(items=[item]),
    )


def _client(conflicts=(), count=None, jobs=JOBS, summary=None):
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
    )
    return fake_client(
        fake_task(jobs=jobs, organization=None),
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


def test_conflict_ids_split_by_job():
    conflicts = [
        _conflict(1, "mismatching_label", _ann(11, 3), _ann(21, 4)),
        _conflict(3, "missing_annotation", _ann(21, 4)),
    ]
    client = _client(conflicts)
    result = create_report(client, "http://cvat", 1, 100)

    _, kwargs = client.api_client.quality_api.list_conflicts.call_args
    assert (kwargs["report_id"], kwargs["page_size"], kwargs["sort"]) == (9, 100, "frame")
    first, second = result.conflicts
    assert (first.frame, first.type, first.shape_ids, first.gt_shape_ids) == (
        1,
        "mismatching_label",
        [11],
        [21],
    )
    assert (second.shape_ids, second.gt_shape_ids) == ([], [21])
    assert result.conflicts_truncated is False


@pytest.mark.parametrize("count, truncated", [(5, True), (2, False)])
def test_conflicts_truncated(count, truncated):
    conflicts = [_conflict(f, "missing_annotation", _ann(21, 4)) for f in (1, 3)]
    result = create_report(_client(conflicts, count=count), "http://cvat", 1, 2)
    assert len(result.conflicts) == 2
    assert result.conflicts_truncated is truncated


@pytest.mark.parametrize("conflict_count", [0, 2])
def test_zero_conflicts_skips_listing(conflict_count):
    client = _client(summary=_summary(conflict_count=conflict_count))
    result = create_report(client, "http://cvat", 1, 0)
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
