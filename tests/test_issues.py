import pytest
from fakes import Obj, fake_client, fake_task, shape

from cvat_mcp.services.issues import create_issue, find_job_for_frame, issue_position

GT = Obj(id=1, type="ground_truth", start_frame=0, stop_frame=39)
FIRST = Obj(id=2, type="annotation", start_frame=0, stop_frame=19)
SECOND = Obj(id=3, type="annotation", start_frame=20, stop_frame=39)


def test_issue_position_from_object_or_bbox():
    assert issue_position([1, 2, 3, 4], None) == [1, 2, 3, 4]
    assert issue_position(None, [1, 2, 3, 4]) == [1, 2, 3, 4]


@pytest.mark.parametrize("args", [([1, 2, 3, 4], [1, 2, 3, 4]), (None, None)])
def test_issue_position_needs_exactly_one(args):
    with pytest.raises(ValueError, match="exactly one"):
        issue_position(*args)


@pytest.mark.parametrize("bbox", [[3, 2, 1, 4], [1, 2, 3], [1, 4, 3, 2]])
def test_issue_position_rejects_bad_bbox(bbox):
    with pytest.raises(ValueError, match="x1 < x2"):
        issue_position(None, bbox)


def test_find_job_for_frame_skips_ground_truth():
    assert find_job_for_frame([GT, FIRST, SECOND], 25) is SECOND
    assert find_job_for_frame([GT, FIRST, SECOND], 5) is FIRST


class FakeIssues:
    def __init__(self):
        self.created = []

    def create(self, request):
        self.created.append(request)
        return Obj(id=77, frame=request.frame, position=request.position)


def _client(issues):
    task = fake_task([shape(11, 1, 25, [10, 20, 110, 220])], jobs=[GT, FIRST, SECOND], task_id=4)
    jobs = {j.id: Obj(task_id=4, **vars(j)) for j in (GT, FIRST, SECOND)}
    return fake_client(task, jobs=Obj(retrieve=jobs.__getitem__), issues=issues)


def test_frame_in_other_job_names_the_right_job():
    issues = FakeIssues()
    with pytest.raises(ValueError, match=r"job 2 \(frames 0\.\.19\)"):
        create_issue(_client(issues), "http://cvat", 3, 5, "msg", None, [1, 2, 3, 4])
    assert not issues.created


def test_frame_outside_task():
    issues = FakeIssues()
    with pytest.raises(ValueError, match=r"0\.\.39"):
        create_issue(_client(issues), "http://cvat", 3, 40, "msg", None, [1, 2, 3, 4])
    assert not issues.created


def test_object_not_on_frame():
    issues = FakeIssues()
    with pytest.raises(ValueError, match="not on frame"):
        create_issue(_client(issues), "http://cvat", 3, 26, "msg", 11, None)
    assert not issues.created


def test_issue_on_object_box():
    issues = FakeIssues()
    ref = create_issue(_client(issues), "http://cvat", 3, 25, "missing label", 11, None)
    (request,) = issues.created
    assert (request.job, request.frame, request.message) == (3, 25, "missing label")
    assert request.position == [10, 20, 110, 220]
    assert ref.issue_id == 77 and ref.url == "http://cvat/tasks/4/jobs/3?frame=25"
