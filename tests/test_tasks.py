from unittest.mock import Mock

import pytest
from fakes import Obj, annotation_job, fake_client, fake_task

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.tasks import (
    JobSummary,
    clamp_limit,
    count_objects_by_label,
    get_task_overview,
    list_jobs,
    list_tasks,
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


def _job(job_id, assignee, issues, start=0, stop=9):
    return annotation_job(
        job_id,
        start,
        stop,
        task_id=4,
        stage="validation",
        state="new",
        assignee=assignee,
        task_name="cars",
        project_id=None,
        frame_count=stop - start + 1,
        issues=Obj(count=issues),
    )


def _owned_by(organization):
    """tasks_api/projects_api whose retrieve answers an object of that organization."""
    return Obj(retrieve=Mock(return_value=(Obj(organization=organization), None)))


def _jobs_client(organization=None):
    jobs = [_job(12, Obj(username="anna"), 2), _job(11, None, 0, 10, 19)]
    listed = Mock(return_value=(Obj(count=57, results=jobs), None))
    api = Obj(
        jobs_api=Obj(list=listed),
        tasks_api=_owned_by(organization),
        projects_api=_owned_by(organization),
    )
    return Obj(api_client=api), listed


@pytest.mark.parametrize("task_id, project_id", [(4, None), (None, 3), (4, 3)])
def test_list_jobs_in_the_filters_organization(task_id, project_id):
    client, listed = _jobs_client(organization=9)
    list_jobs(client, "http://cvat", None, None, None, task_id, project_id, None, 20)
    assert listed.call_args.kwargs["org_id"] == 9
    if task_id is not None:  # a task decides; its project needs no second request
        client.api_client.projects_api.retrieve.assert_not_called()


def test_list_jobs_without_task_or_project_asks_for_no_organization():
    client, listed = _jobs_client(organization=9)
    list_jobs(client, "http://cvat", "validation", None, None, None, None, None, 20)
    assert "org_id" not in listed.call_args.kwargs
    client.api_client.tasks_api.retrieve.assert_not_called()


def test_list_tasks_in_the_projects_organization():
    listed = Mock(return_value=(Obj(count=0, results=[]), None))
    client = Obj(api_client=Obj(tasks_api=Obj(list=listed), projects_api=_owned_by(9)))
    list_tasks(client, 3, None, 20)
    assert listed.call_args.kwargs == {"page_size": 20, "sort": "-id", "project_id": 3, "org_id": 9}


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


@pytest.mark.parametrize(
    "args, sent",
    [
        (
            ("validation", None, None, None, 3, "annotation"),
            {"stage": "validation", "project_id": 3, "type": "annotation"},
        ),
        (  # type None: no type filter, so every job type is listed
            (None, "new", "anna", 4, None, None),
            {"state": "new", "assignee": "anna", "task_id": 4},
        ),
    ],
)
def test_list_jobs_sends_only_given_filters(args, sent):
    client, listed = _jobs_client()
    list_jobs(client, "http://cvat", *args, 20)
    listed.assert_called_once_with(page_size=20, sort="-id", **sent)


def test_list_jobs_job_without_task_name():
    client, listed = _jobs_client()
    listed.return_value[0].results[0].task_name = None
    result = list_jobs(client, "http://cvat", None, None, None, None, None, None, 20)
    assert result.jobs[0].task_name is None


def test_list_jobs_refuses_empty_assignee():
    client, listed = _jobs_client()
    with pytest.raises(InvalidArgument, match="unassigned"):
        list_jobs(client, "http://cvat", None, None, "", None, None, None, 20)
    listed.assert_not_called()


@pytest.mark.parametrize("limit, page_size", [(0, 1), (500, 100), (20, 20)])
def test_list_jobs_clamps_limit_and_reports_total(limit, page_size):
    client, listed = _jobs_client()
    result = list_jobs(client, "http://cvat", None, None, None, None, None, None, limit)
    assert listed.call_args.kwargs["page_size"] == page_size and result.total == 57
