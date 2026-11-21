from unittest.mock import Mock

import pytest
from fakes import Obj, fake_client

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.job_edits import update_job

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
