from unittest.mock import Mock

import pytest
from fakes import Obj, fake_client, fake_task, shape

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.issues import check_bbox, create_issue, find_job_for_frame

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


def test_issue_on_object_box():
    issues = Mock()
    issues.create.return_value = Obj(id=77)
    ref = create_issue(_client(issues), "http://cvat", 3, 25, "missing label", 11, None)
    request = issues.create.call_args.args[0]
    assert (request.job, request.frame, request.message) == (3, 25, "missing label")
    assert request.position == [10, 20, 110, 220]
    assert ref.issue_id == 77 and ref.url == "http://cvat/tasks/4/jobs/3?frame=25"
