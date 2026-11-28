from unittest.mock import Mock

import pytest
from fakes import Obj, fake_client, fake_task

from cvat_mcp.services.tasks import (
    JobSummary,
    clamp_limit,
    count_objects_by_label,
    get_task_overview,
    list_jobs,
)


def test_clamp_limit():
    assert clamp_limit(0) == 1
    assert clamp_limit(-5) == 1
    assert clamp_limit(1000) == 100
    assert clamp_limit(20) == 20


def test_count_objects_by_label_empty_keeps_all_labels():
    assert count_objects_by_label([], {1: "car", 2: "bus"}) == {"car": 0, "bus": 0}


def test_count_objects_by_label():
    assert count_objects_by_label([1, 1, 2], {1: "car", 2: "bus"}) == {"car": 2, "bus": 1}


class _NoProjectAccess:
    def retrieve(self, _id):
        raise AssertionError("must not request the project: the user may lack access")


def test_overview_uses_task_project_name_and_counts_tags():
    one = Obj(label_id=1)
    task = fake_task(
        [one], tracks=[one], tags=[one], name="t", project_id=2, project_name="Traffic", task_id=3
    )
    overview = get_task_overview(fake_client(task, projects=_NoProjectAccess()), 3)
    assert overview.project_name == "Traffic"
    assert overview.objects_per_label == {"car": 3, "person": 0}
    assert overview.total_objects == 3


def _job(id, assignee, issues, start=0, stop=9):  # pylint: disable=redefined-builtin
    return Obj(
        id=id,
        type="annotation",
        start_frame=start,
        stop_frame=stop,
        stage="validation",
        state="new",
        assignee=assignee,
        task_id=4,
        task_name="cars",
        project_id=None,
        frame_count=stop - start + 1,
        issues=Obj(count=issues),
    )


def _jobs_client():
    jobs = [_job(12, Obj(username="anna"), 2), _job(11, None, 0, 10, 19)]
    listed = Mock(return_value=(Obj(count=57, results=jobs), None))
    return Obj(api_client=Obj(jobs_api=Obj(list=listed))), listed


def test_list_jobs_summaries():
    client, _ = _jobs_client()
    result = list_jobs(client, "http://cvat", None, None, None, None, None, "annotation", 20)
    assert result.total == 57
    assert result.jobs[0] == JobSummary(
        id=12,
        type="annotation",
        start_frame=0,
        stop_frame=9,
        stage="validation",
        state="new",
        assignee="anna",
        task_id=4,
        task_name="cars",
        project_id=None,
        frame_count=10,
        issues=2,
        url="http://cvat/tasks/4/jobs/12?frame=0",
    )
    assert result.jobs[1].assignee is None
    assert result.jobs[1].url == "http://cvat/tasks/4/jobs/11?frame=10"


def test_list_jobs_sends_only_given_filters():
    client, listed = _jobs_client()
    list_jobs(client, "http://cvat", "validation", None, None, None, 3, "annotation", 20)
    listed.assert_called_once_with(
        page_size=20, sort="-id", stage="validation", project_id=3, type="annotation"
    )


def test_list_jobs_type_none_sends_no_type():
    client, listed = _jobs_client()
    list_jobs(client, "http://cvat", None, "new", "anna", 4, None, None, 20)
    assert listed.call_args.kwargs == {
        "page_size": 20,
        "sort": "-id",
        "state": "new",
        "assignee": "anna",
        "task_id": 4,
    }


@pytest.mark.parametrize("limit, page_size", [(0, 1), (500, 100), (20, 20)])
def test_list_jobs_clamps_limit_and_reports_total(limit, page_size):
    client, listed = _jobs_client()
    result = list_jobs(client, "http://cvat", None, None, None, None, None, None, limit)
    assert listed.call_args.kwargs["page_size"] == page_size and result.total == 57
