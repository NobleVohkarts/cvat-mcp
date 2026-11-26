from unittest.mock import Mock

import pytest
from cvat_sdk.api_client.exceptions import ApiException
from cvat_sdk.core.exceptions import BackgroundRequestException
from cvat_sdk.core.proxies.tasks import ResourceType
from fakes import Obj, cloud_storage, fake_client, fake_cloudstorages_api

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.task_edits import (
    AttributeSpec,
    LabelSpec,
    create_task,
    delete_task,
    label_requests,
    update_task,
)

URL = "https://example.com/a.jpg"
COLOR = AttributeSpec(name="color", input_type="select", values=["red", "blue"])


def _with_attr(**kw):
    return [LabelSpec(name="car", attributes=[AttributeSpec(**kw)])]


def test_label_requests_build_cvat_labels():
    (car,) = label_requests([LabelSpec(name="car", type="rectangle", attributes=[COLOR])])
    assert (car.name, str(car.type)) == ("car", "rectangle")
    (color,) = car.attributes
    assert (color.name, color.input_type.value, color.values) == (
        "color",
        "select",
        ["red", "blue"],
    )


def test_checkbox_and_text_get_default_values():
    specs = [
        AttributeSpec(name="parked", input_type="checkbox"),
        AttributeSpec(name="plate", input_type="text"),
    ]
    (label,) = label_requests([LabelSpec(name="car", attributes=specs)])
    assert [a.values for a in label.attributes] == [["false"], [""]]


@pytest.mark.parametrize(
    "specs, existing, match",
    [
        ([], (), "at least one"),
        ([LabelSpec(name="car"), LabelSpec(name="car")], (), "twice"),
        ([LabelSpec(name="car")], ("car",), "already has"),
        ([LabelSpec(name=" ")], (), "name"),
        (_with_attr(name="c", input_type="select"), (), "'c'"),
        (_with_attr(name="n", input_type="number", values=["0", "10"]), (), "'n'"),
        (_with_attr(name="n", input_type="number", values=["0", "10", "3"]), (), "'n'"),
        (_with_attr(name="p", input_type="checkbox", values=["yes"]), (), "'p'"),
    ],
)
def test_label_requests_refuse(specs, existing, match):
    with pytest.raises(InvalidArgument, match=match):
        label_requests(specs, existing)


JOB = Obj(
    id=8,
    type="annotation",
    start_frame=0,
    stop_frame=2,
    stage="annotation",
    state="new",
    assignee=None,
)


def _create_client(upload_error=None, remove_error=None, storages=(cloud_storage(1, "cvat"),)):
    task = Obj(id=5, name="Scratch", size=3, get_jobs=lambda: [JOB])
    task.upload_data = Mock(side_effect=upload_error)
    task.remove = Mock(side_effect=remove_error)
    client = Obj(
        tasks=Obj(create=Mock(return_value=task)),
        api_client=Obj(cloudstorages_api=fake_cloudstorages_api(storages)),
    )
    return client, task


def _create(client, **kw):
    args = {
        "name": "Scratch",
        "labels": [LabelSpec(name="car")],
        "project_id": None,
        "urls": None,
        "cloud_storage_id": None,
        "cloud_path": None,
        "segment_size": None,
    } | kw
    return create_task(client, "http://cvat/", **args)


def test_create_from_urls_returns_jobs():
    urls = ["https://example.com/1.jpg", "https://example.com/2.jpg"]
    client, task = _create_client()
    result = _create(client, urls=urls)
    (spec,), _ = client.tasks.create.call_args
    sent = spec.to_dict()
    assert sent["name"] == "Scratch" and [l["name"] for l in sent["labels"]] == ["car"]
    assert "project_id" not in sent and "segment_size" not in sent
    task.upload_data.assert_called_once_with(
        resources=urls,
        resource_type=ResourceType.REMOTE,
        params={"sorting_method": "natural"},
        wait_for_completion=True,
        status_check_period=1,
    )
    assert (result.task_id, result.name, result.frame_count) == (5, "Scratch", 3)
    assert [(j.id, j.start_frame, j.stop_frame) for j in result.jobs] == [(8, 0, 2)]
    assert result.url == "http://cvat/tasks/5"


def test_create_with_segment_size():
    client, _ = _create_client()
    _create(client, urls=["https://example.com/b.jpg"], segment_size=20)
    (spec,), _ = client.tasks.create.call_args
    assert spec.to_dict()["segment_size"] == 20


def test_create_from_urls_uses_remote():
    client, task = _create_client()
    _create(client, urls=["https://example.com/a.jpg"])
    _, kwargs = task.upload_data.call_args
    assert kwargs["resource_type"] == ResourceType.REMOTE
    assert kwargs["resources"] == ["https://example.com/a.jpg"]


def test_create_in_project_sends_no_labels():
    client, _ = _create_client()
    _create(client, labels=None, project_id=9, urls=["https://example.com/a.jpg"])
    (spec,), _ = client.tasks.create.call_args
    assert spec.to_dict()["project_id"] == 9 and "labels" not in spec.to_dict()


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"project_id": 9, "urls": [URL]}, "one of labels or project_id"),
        ({"labels": None, "urls": [URL]}, "one of labels or project_id"),
        (
            {"urls": [URL], "cloud_storage_id": 1, "cloud_path": "x/"},
            "one of urls or cloud_storage_id",
        ),
        ({}, "one of urls or cloud_storage_id \\+ cloud_path"),
        ({"cloud_storage_id": 1}, "cloud_storage_id and cloud_path together"),
        ({"cloud_path": "street/"}, "cloud_storage_id and cloud_path together"),
        ({"urls": ["ftp://h/x.jpg"]}, "http"),
        ({"urls": ["file:///x"]}, "http"),
        ({"name": " ", "urls": [URL]}, "name"),
    ],
)
def test_create_refusals(kw, match):
    client, _ = _create_client()
    with pytest.raises(InvalidArgument, match=match):
        _create(client, **kw)
    client.tasks.create.assert_not_called()


@pytest.mark.parametrize("remove_error", [None, RuntimeError("remove failed")])
def test_failed_upload_removes_the_task(remove_error):
    client, task = _create_client(BackgroundRequestException("bad image"), remove_error)
    with pytest.raises(BackgroundRequestException, match="bad image"):
        _create(client, urls=["https://example.com/a.jpg"])
    task.remove.assert_called_once()


def _uploaded(task):
    _, kwargs = task.upload_data.call_args
    return kwargs["resources"], kwargs["params"]


def test_create_from_bucket_folder():
    client, task = _create_client()
    result = _create(client, cloud_storage_id=1, cloud_path="street/")
    task.upload_data.assert_called_once_with(
        resources=["street/"],  # CVAT lists only this prefix
        resource_type=ResourceType.SHARE,
        params={
            "sorting_method": "natural",
            "cloud_storage_id": 1,
            "use_cache": True,
        },
        wait_for_completion=True,
        status_check_period=1,
    )
    assert result.cloud_storage_id == 1
    client.api_client.cloudstorages_api.list.assert_not_called()
    client.api_client.cloudstorages_api.create.assert_not_called()


def test_folder_path_without_slash_is_a_folder():
    client, task = _create_client()
    _create(client, cloud_storage_id=1, cloud_path="street")
    assert _uploaded(task)[0] == ["street/"]


def test_create_from_bucket_object():
    client, task = _create_client()
    _create(client, cloud_storage_id=1, cloud_path="street/a.jpg")
    resources, params = _uploaded(task)
    assert resources == ["street/a.jpg"]
    assert "filename_pattern" not in params


def test_create_from_glob():
    client, task = _create_client()
    _create(client, cloud_storage_id=1, cloud_path="street/*.jpg")
    resources, params = _uploaded(task)
    assert resources == [] and params["filename_pattern"] == "street/*.jpg"


def test_background_failure_names_the_path():
    client, task = _create_client(BackgroundRequestException("No media data found"))
    with pytest.raises(
        InvalidArgument, match="No media data found.*cloud_path 'nothing/'"
    ) as error:
        _create(client, cloud_storage_id=1, cloud_path="nothing/")
    assert "update_cloud_storage" in str(error.value)
    task.remove.assert_called_once()


@pytest.mark.parametrize("status", [403, 404])
def test_unknown_storage_refused_before_the_task(status):
    client, _ = _create_client()
    client.api_client.cloudstorages_api.retrieve = Mock(side_effect=ApiException(status=status))
    with pytest.raises(InvalidArgument, match="no cloud storage #99.*list_cloud_storages"):
        _create(client, cloud_storage_id=99, cloud_path="street/")
    client.tasks.create.assert_not_called()


@pytest.mark.parametrize("path", ["highway/", "", "*.jpg"])
def test_path_outside_the_storage_prefix_refused(path):
    client, _ = _create_client(storages=[cloud_storage(1, "cvat", "prefix=street%2F")])
    with pytest.raises(InvalidArgument, match="covers only 'street/'.*list_cloud_storages"):
        _create(client, cloud_storage_id=1, cloud_path=path)
    client.tasks.create.assert_not_called()


@pytest.mark.parametrize("path", ["street/", "street", "street/day/*.jpg", "street/a.jpg"])
def test_path_inside_the_storage_prefix_used(path):
    client, task = _create_client(storages=[cloud_storage(1, "cvat", "prefix=street/")])
    _create(client, cloud_storage_id=1, cloud_path=path)
    task.upload_data.assert_called_once()


def test_organization_storage_refused_before_the_task():
    client, _ = _create_client(storages=[cloud_storage(1, "cvat", org=5)])
    with pytest.raises(InvalidArgument, match="cloud storage #1 belongs to an organization"):
        _create(client, cloud_storage_id=1, cloud_path="street/")
    client.tasks.create.assert_not_called()


def test_unavailable_storage_points_at_its_keys():
    unavailable = ApiException(status=400)
    unavailable.body = b"The specified cloud storage 'x' is not available."
    client, task = _create_client(unavailable)
    with pytest.raises(InvalidArgument, match="update_cloud_storage"):
        _create(client, cloud_storage_id=1, cloud_path="street/")
    task.remove.assert_called_once()
    client.api_client.cloudstorages_api.destroy.assert_not_called()


ADMIN = Obj(id=1, username="admin")
CAR_LABEL = Obj(id=1, name="car")


def _task_client(owner=ADMIN, users=()):
    task = Obj(id=5, name="Scratch", size=3, owner=owner, project_id=None, assignee=None)
    task.get_labels = lambda: [CAR_LABEL]
    task.update, task.remove = Mock(), Mock()
    users = Obj(retrieve_current_user=lambda: ADMIN, list=Mock(return_value=list(users)))
    return fake_client(task, users=users), task


def _update(client, **kw):
    args = {"name": None, "assignee": None, "project_id": None, "add_labels": None} | kw
    return update_task(client, 5, **args)


def _sent(task):
    (request,), _ = task.update.call_args
    return request.to_dict()


def test_update_sends_only_given_fields():
    client, task = _task_client()
    result = _update(client, name="New")
    assert _sent(task) == {"name": "New"}
    assert (result.task_id, result.labels) == (5, ["car"])


def test_update_assignee_by_username():
    client, task = _task_client(users=[Obj(id=7, username="bob")])
    _update(client, assignee="bob")
    client.users.list.assert_called_once_with(username="bob")
    assert _sent(task) == {"assignee_id": 7}


def test_update_empty_assignee_unassigns():
    client, task = _task_client()
    _update(client, assignee="")
    client.users.list.assert_not_called()
    assert _sent(task) == {"assignee_id": None}


def test_update_unknown_user():
    client, task = _task_client(users=[])
    with pytest.raises(InvalidArgument, match="no CVAT user 'nobody'"):
        _update(client, assignee="nobody")
    task.update.assert_not_called()


def test_update_add_labels_checks_existing():
    client, task = _task_client()
    with pytest.raises(InvalidArgument, match="already has"):
        _update(client, add_labels=[LabelSpec(name="car")])
    task.update.assert_not_called()


def test_update_add_labels():
    client, task = _task_client()
    _update(client, add_labels=[LabelSpec(name="bus", type="rectangle")])
    assert [l["name"] for l in _sent(task)["labels"]] == ["bus"]


def test_update_needs_a_field():
    client, task = _task_client()
    with pytest.raises(InvalidArgument, match="nothing to change"):
        _update(client)
    task.update.assert_not_called()


def test_delete_own_task():
    client, task = _task_client()
    result = delete_task(client, 5)
    task.remove.assert_called_once()
    assert (result.task_id, result.name, result.frame_count) == (5, "Scratch", 3)


@pytest.mark.parametrize(
    "owner, match", [(Obj(id=2, username="alice"), "alice.*CVAT UI"), (None, "a deleted user")]
)
def test_delete_refuses_foreign_or_ownerless(owner, match):
    client, task = _task_client(owner=owner)
    with pytest.raises(InvalidArgument, match=match):
        delete_task(client, 5)
    task.remove.assert_not_called()


def test_label_names_are_stripped():
    (label,) = label_requests([LabelSpec(name=" bus ")])
    assert label.name == "bus"
    with pytest.raises(InvalidArgument, match="already has"):
        label_requests([LabelSpec(name="car ")], ["car"])


@pytest.mark.parametrize(
    "kw", [{"name": "New", "project_id": 3}, {"assignee": "", "project_id": 3}]
)
def test_update_refuses_project_move_with_other_changes(kw):
    client, task = _task_client()
    with pytest.raises(InvalidArgument, match="project_id alone"):
        _update(client, **kw)
    task.update.assert_not_called()


def test_update_refuses_labels_for_a_task_in_a_project():
    client, task = _task_client()
    task.project_id = 3
    with pytest.raises(InvalidArgument, match="project"):
        _update(client, add_labels=[LabelSpec(name="bus")])
    task.update.assert_not_called()


def test_update_reports_labels_read_back():
    client, task = _task_client()
    labels = iter([[CAR_LABEL], [CAR_LABEL]])  # CVAT kept only car
    task.get_labels = lambda: next(labels)
    result = _update(client, add_labels=[LabelSpec(name="bus")])
    assert result.labels == ["car"]
