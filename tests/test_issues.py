from unittest.mock import Mock

import pytest
from fakes import Obj, fake_client, fake_task, shape

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.issues import (
    check_bbox,
    create_issue,
    delete_issue,
    find_job_for_frame,
    list_issues,
)

GT = Obj(id=1, type="ground_truth", start_frame=0, stop_frame=39)
FIRST = Obj(id=2, type="annotation", start_frame=0, stop_frame=19)
SECOND = Obj(id=3, type="annotation", start_frame=20, stop_frame=39)
ON_25 = shape(11, 1, 25, [10, 20, 110, 220])


@pytest.mark.parametrize("bbox", [[3, 2, 1, 4], [1, 4, 3, 2]])
def test_check_bbox_rejects_inverted(bbox):
    with pytest.raises(InvalidArgument, match="x1 < x2"):
        check_bbox(bbox)


def test_find_job_for_frame_skips_ground_truth():
    assert find_job_for_frame([GT, FIRST, SECOND], 25) is SECOND
    assert find_job_for_frame([GT, FIRST, SECOND], 5) is FIRST


def _client(issues):
    task = fake_task(jobs=[GT, FIRST, SECOND], task_id=4)
    jobs = {
        j.id: Obj(task_id=4, get_annotations=lambda: Obj(shapes=[ON_25]), **vars(j))
        for j in (GT, FIRST, SECOND)
    }
    return fake_client(task, jobs=Obj(retrieve=jobs.__getitem__), issues=issues)


@pytest.mark.parametrize(
    "job_id, frame, message, object_id, bbox, match",
    [
        (3, 5, "msg", None, [1, 2, 3, 4], r"job 2 \(frames 0\.\.19\)"),  # frame in another job
        (1, 5, "msg", None, [1, 2, 3, 4], r"job 2 \(frames 0\.\.19\)"),  # ground-truth job
        (3, 26, "msg", 11, None, "not on frame"),
        (3, 25, "msg", 11, [1, 2, 3, 4], "exactly one"),
        (3, 25, "msg", None, None, "exactly one"),
        (3, 25, " ", None, [1, 2, 3, 4], "message"),
    ],
)
def test_rejected_without_creating(job_id, frame, message, object_id, bbox, match):
    issues = Mock()
    with pytest.raises(InvalidArgument, match=match):
        create_issue(_client(issues), "http://cvat", job_id, frame, message, object_id, bbox)
    issues.create.assert_not_called()


def test_argument_errors_need_no_requests():
    client = Obj(jobs=Obj(retrieve=Mock(side_effect=AssertionError("no I/O"))))
    with pytest.raises(InvalidArgument, match="exactly one"):
        create_issue(client, "http://cvat", 3, 25, "msg", None, None)


def test_url_without_double_slash():
    issues = Mock()
    issues.create.return_value = Obj(id=77)
    ref = create_issue(_client(issues), "http://cvat/", 3, 25, "m", None, [1, 2, 3, 4])
    assert ref.url == "http://cvat/tasks/4/jobs/3?frame=25"


def test_issue_on_object_box():
    issues = Mock()
    issues.create.return_value = Obj(id=77)
    ref = create_issue(_client(issues), "http://cvat", 3, 25, "missing label", 11, None)
    request = issues.create.call_args.args[0]
    assert (request.job, request.frame, request.message) == (3, 25, "missing label")
    assert request.position == [10, 20, 110, 220]
    assert ref.issue_id == 77 and ref.url == "http://cvat/tasks/4/jobs/3?frame=25"


ME = Obj(id=5, username="reviewer")
ANNA = Obj(id=6, username="anna")


def _issue(id, job, frame, owner, count=1, **kw):  # pylint: disable=redefined-builtin
    return Obj(
        id=id,
        job=job,
        frame=frame,
        position=[1.0, 2.0, 3.0, 4.0],
        resolved=False,
        owner=owner,
        comments=Obj(count=count),
        **kw,
    )


def _comment(id, issue, message):  # pylint: disable=redefined-builtin
    return Obj(id=id, issue=issue, message=message)


@pytest.fixture
def listed():
    """Issue and comment repos for a fake client; records every call as (repo, filters)."""
    calls = []
    data = {
        "issues": [_issue(41, 3, 25, ME, count=2), _issue(40, 2, 5, ANNA), _issue(42, 3, 25, None)],
        "comments": [
            _comment(102, 41, "reply"),
            _comment(101, 41, "box misses the car"),
            _comment(100, 40, "label is wrong"),
            _comment(103, 42, "missing person"),
        ],
    }

    def repo(name):
        def list_(**kwargs):
            calls.append((name, kwargs))
            rows = data[name]
            if "job_id" in kwargs:  # comments filtered by job, as CVAT does
                jobs = {i.id: i.job for i in data["issues"]}
                rows = [c for c in rows if jobs[c.issue] == kwargs["job_id"]]
            return rows

        return Obj(list=list_)

    return data, calls, repo


def _list_client(listed, organization=None):
    repo = listed[2]
    return fake_client(
        fake_task(task_id=4, size=40, organization=organization),
        issues=repo("issues"),
        comments=repo("comments"),
        users=Obj(retrieve_current_user=lambda: ME),
    )


def test_list_issues_messages_authors_and_own(listed):
    result = list_issues(_list_client(listed), 4, None)
    assert [
        (i.issue_id, i.frame, i.message, i.author, i.own, i.replies) for i in result.issues
    ] == [
        (40, 5, "label is wrong", "anna", False, 0),
        (41, 25, "box misses the car", "reviewer", True, 1),
        (42, 25, "missing person", None, False, 0),
    ]


def test_list_issues_one_comments_request_per_job_with_org_and_frame(listed):
    calls = listed[1]
    list_issues(_list_client(listed, organization=9), 4, 25)
    assert calls[0] == ("issues", {"org_id": 9, "task_id": 4, "frame_id": 25})
    assert sorted(calls[1:], key=lambda c: c[1]["job_id"]) == [
        ("comments", {"org_id": 9, "job_id": 2, "frame_id": 25}),
        ("comments", {"org_id": 9, "job_id": 3, "frame_id": 25}),
    ]


def test_list_issues_without_issues_asks_for_no_comments(listed):
    data, calls, _ = listed
    data["issues"] = []
    assert list_issues(_list_client(listed), 4, 7).issues == []
    assert [c[0] for c in calls] == ["issues"]


def test_list_issues_checks_frame(listed):
    with pytest.raises(InvalidArgument, match="0..39"):
        list_issues(_list_client(listed), 4, 40)
    assert listed[1] == []


def _delete_client(issue):
    return Obj(issues=Obj(retrieve=lambda _id: issue), users=Obj(retrieve_current_user=lambda: ME))


def test_delete_own_issue():
    remove = Mock()
    issue = _issue(41, 3, 25, ME, remove=remove)
    assert delete_issue(_delete_client(issue), 41).model_dump() == {
        "issue_id": 41,
        "job_id": 3,
        "frame": 25,
    }
    remove.assert_called_once_with()


@pytest.mark.parametrize("owner, author", [(ANNA, "anna"), (None, "a deleted user")])
def test_delete_refuses_foreign_issue(owner, author):
    remove = Mock()
    issue = _issue(40, 2, 5, owner, remove=remove)
    with pytest.raises(InvalidArgument, match=f"filed by {author}.*CVAT UI"):
        delete_issue(_delete_client(issue), 40)
    remove.assert_not_called()
