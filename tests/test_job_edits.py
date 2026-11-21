from unittest.mock import Mock

import pytest
from fakes import Obj, fake_client

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.job_edits import create_gt_job, delete_gt_job, update_job

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
    return Obj(**(defaults | kw))


def _client(job, users=()):
    users = Obj(retrieve_current_user=lambda: ADMIN, list=Mock(return_value=list(users)))
    jobs_api = Obj(partial_update=Mock(return_value=(job, None)))
    return fake_client(Obj(id=5), users=users, api_client=Obj(jobs_api=jobs_api))


def _update(client, **kw):
    return update_job(client, 20, **({"assignee": None, "stage": None, "state": None} | kw))


def _sent(client):
    partial_update = client.api_client.jobs_api.partial_update
    (job_id,), kwargs = partial_update.call_args
    assert job_id == 20
    return kwargs["patched_job_write_request"].to_dict()


def test_update_job_sends_only_given_fields():
    client = _client(_job())
    result = _update(client, stage="validation")
    assert _sent(client) == {"stage": "validation"}
    assert (result.id, result.type) == (20, "annotation")


def test_update_job_assignee_by_username():
    client = _client(_job(), users=[Obj(id=7, username="bob")])
    _update(client, assignee="bob")
    assert _sent(client) == {"assignee": 7}


def test_update_job_empty_assignee_unassigns():
    client = _client(_job())
    _update(client, assignee="")
    client.users.list.assert_not_called()
    assert _sent(client) == {"assignee": None}


def test_update_job_unknown_user():
    client = _client(_job())
    with pytest.raises(InvalidArgument, match="no CVAT user"):
        _update(client, assignee="nobody")
    client.api_client.jobs_api.partial_update.assert_not_called()


def test_update_job_needs_a_field():
    client = _client(_job())
    with pytest.raises(InvalidArgument, match="nothing to change"):
        _update(client)
    client.api_client.jobs_api.partial_update.assert_not_called()


def _gt_client(included=(0, 2)):
    jobs_api = Obj(
        create=Mock(return_value=(Obj(id=30), None)),
        retrieve_data_meta=Mock(return_value=(Obj(included_frames=list(included)), None)),
    )
    client = fake_client(Obj(id=5, size=3), api_client=Obj(jobs_api=jobs_api))
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


def _delete_client(job_type="ground_truth", owner=ADMIN):
    job = _job(id=30, type=job_type, frame_count=2)
    task = Obj(id=5, owner=owner)
    jobs_api = Obj(destroy=Mock())
    client = fake_client(
        task,
        jobs=Obj(retrieve=lambda _id: job),
        users=Obj(retrieve_current_user=lambda: ADMIN),
        api_client=Obj(jobs_api=jobs_api),
    )
    return client, jobs_api


def test_delete_gt_job():
    client, jobs_api = _delete_client()
    result = delete_gt_job(client, 30)
    jobs_api.destroy.assert_called_once_with(30)
    assert (result.job_id, result.task_id, result.frame_count) == (30, 5, 2)


def test_delete_refuses_annotation_job():
    client, jobs_api = _delete_client(job_type="annotation")
    with pytest.raises(InvalidArgument, match="segments.*delete the task"):
        delete_gt_job(client, 30)
    jobs_api.destroy.assert_not_called()


def test_delete_refuses_foreign_task():
    client, jobs_api = _delete_client(owner=Obj(id=2, username="alice"))
    with pytest.raises(InvalidArgument, match="alice.*CVAT UI"):
        delete_gt_job(client, 30)
    jobs_api.destroy.assert_not_called()
