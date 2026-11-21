from unittest.mock import Mock

import pytest
from fakes import Obj, fake_client

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.job_edits import create_gt_job, update_job

ADMIN = Obj(id=1, username="admin")


def _job(**kw):
    defaults = {
        "id": 20,
        "task_id": 5,
        "type": "annotation",
        "start_frame": 0,
        "stop_frame": 2,
        "stage": "annotation",
        "state": "new",
        "assignee": None,
    }
    job = Obj(**(defaults | kw))
    job.update = Mock()
    return job


def _client(job, users=()):
    users = Obj(retrieve_current_user=lambda: ADMIN, list=Mock(return_value=list(users)))
    return fake_client(Obj(id=5), jobs=Obj(retrieve=lambda _id: job), users=users)


def _update(client, **kw):
    return update_job(client, 20, **({"assignee": None, "stage": None, "state": None} | kw))


def _sent(job):
    (request,), _ = job.update.call_args
    return request.to_dict()


def test_update_job_sends_only_given_fields():
    job = _job()
    result = _update(_client(job), stage="validation")
    assert _sent(job) == {"stage": "validation"}
    assert (result.id, result.type) == (20, "annotation")


def test_update_job_assignee_by_username():
    job = _job()
    client = _client(job, users=[Obj(id=7, username="bob")])
    _update(client, assignee="bob")
    assert _sent(job) == {"assignee": 7}


def test_update_job_empty_assignee_unassigns():
    job = _job()
    client = _client(job)
    _update(client, assignee="")
    client.users.list.assert_not_called()
    assert _sent(job) == {"assignee": None}


def test_update_job_unknown_user():
    job = _job()
    with pytest.raises(InvalidArgument, match="no CVAT user"):
        _update(_client(job), assignee="nobody")
    job.update.assert_not_called()


def test_update_job_needs_a_field():
    job = _job()
    with pytest.raises(InvalidArgument, match="nothing to change"):
        _update(_client(job))
    job.update.assert_not_called()


def _gt_client(included=(0, 2)):
    jobs_api = Obj(create=Mock(return_value=(Obj(id=30), None)), destroy=Mock())
    gt = Obj(id=30, get_meta=lambda: Obj(included_frames=list(included)))
    client = fake_client(
        Obj(id=5, size=3), jobs=Obj(retrieve=lambda _id: gt), api_client=Obj(jobs_api=jobs_api)
    )
    return client, jobs_api


def _create(client, **kw):
    return create_gt_job(client, 5, **({"frame_count": None, "frames": None, "seed": None} | kw))


def _created(jobs_api):
    (request,), _ = jobs_api.create.call_args
    return {
        k: str(v) if k in ("type", "frame_selection_method") else v
        for k, v in request.to_dict().items()
    }


def test_create_gt_job_random():
    client, jobs_api = _gt_client()
    result = _create(client, frame_count=2, seed=1)
    assert _created(jobs_api) == {
        "type": "ground_truth",
        "task_id": 5,
        "frame_selection_method": "random_uniform",
        "frame_count": 2,
        "random_seed": 1,
    }
    assert (result.job_id, result.task_id, result.frames) == (30, 5, [0, 2])


def test_create_gt_job_manual():
    client, jobs_api = _gt_client()
    _create(client, frames=[2, 0])
    assert _created(jobs_api) == {
        "type": "ground_truth",
        "task_id": 5,
        "frame_selection_method": "manual",
        "frames": [0, 2],
    }


@pytest.mark.parametrize(
    "kw, match",
    [
        ({}, "one of"),
        ({"frame_count": 2, "frames": [0]}, "one of"),
        ({"frame_count": 0}, r"1\.\.3"),
        ({"frame_count": 4}, r"1\.\.3"),
        ({"frames": [0, 0]}, "twice"),
        ({"frames": [3]}, "outside"),
        ({"frames": [0], "seed": 1}, "seed"),
    ],
)
def test_create_gt_job_refusals(kw, match):
    client, jobs_api = _gt_client()
    with pytest.raises(InvalidArgument, match=match):
        _create(client, **kw)
    jobs_api.create.assert_not_called()
