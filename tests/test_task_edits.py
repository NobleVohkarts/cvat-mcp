from unittest.mock import Mock

import pytest
from cvat_sdk.api_client.exceptions import ApiException
from cvat_sdk.core.exceptions import BackgroundRequestException
from cvat_sdk.core.proxies.tasks import ResourceType
from fakes import Obj, fake_client, fake_cloudstorages_api

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.task_edits import (
    AttributeSpec,
    LabelSpec,
    create_task,
    delete_task,
    label_requests,
    resolve_files,
    update_task,
)

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


def _create_client(upload_error=None, remove_error=None):
    task = Obj(id=5, name="Scratch", size=3, get_jobs=lambda: [JOB])
    task.upload_data = Mock(side_effect=upload_error)
    task.remove = Mock(side_effect=remove_error)
    client = Obj(
        tasks=Obj(create=Mock(return_value=task)),
        api_client=Obj(cloudstorages_api=fake_cloudstorages_api()),  # no storage requests expected
    )
    return client, task


def _create(client, workdir, **kw):
    args = {
        "name": "Scratch",
        "labels": [LabelSpec(name="car")],
        "project_id": None,
        "files": None,
        "urls": None,
        "cloud_storage_id": None,
        "cloud_path": None,
        "segment_size": None,
    } | kw
    return create_task(client, "http://cvat/", workdir, **args)


def test_create_from_files_uploads_sorted_and_returns_jobs(tmp_path):
    (tmp_path / "a").mkdir()
    for name in ("a/2.jpg", "a/1.jpg", "b.jpg"):
        (tmp_path / name).write_bytes(b"x")
    client, task = _create_client()
    result = _create(client, tmp_path, files=["a", "b.jpg"])
    (spec,), _ = client.tasks.create.call_args
    sent = spec.to_dict()
    assert sent["name"] == "Scratch" and [l["name"] for l in sent["labels"]] == ["car"]
    assert "project_id" not in sent and "segment_size" not in sent
    task.upload_data.assert_called_once_with(
        resources=[tmp_path / "a/1.jpg", tmp_path / "a/2.jpg", tmp_path / "b.jpg"],
        resource_type=ResourceType.LOCAL,
        params={"sorting_method": "natural"},
        wait_for_completion=True,
        status_check_period=1,
    )
    assert (result.task_id, result.name, result.frame_count) == (5, "Scratch", 3)
    assert [(j.id, j.start_frame, j.stop_frame) for j in result.jobs] == [(8, 0, 2)]
    assert result.url == "http://cvat/tasks/5"


def test_create_with_segment_size(tmp_path):
    (tmp_path / "b.jpg").write_bytes(b"x")
    client, _ = _create_client()
    _create(client, tmp_path, files=["b.jpg"], segment_size=20)
    (spec,), _ = client.tasks.create.call_args
    assert spec.to_dict()["segment_size"] == 20


def test_create_from_urls_uses_remote(tmp_path):
    client, task = _create_client()
    _create(client, tmp_path, urls=["https://example.com/a.jpg"])
    _, kwargs = task.upload_data.call_args
    assert kwargs["resource_type"] == ResourceType.REMOTE
    assert kwargs["resources"] == ["https://example.com/a.jpg"]


def test_create_in_project_sends_no_labels(tmp_path):
    client, _ = _create_client()
    _create(client, tmp_path, labels=None, project_id=9, urls=["https://example.com/a.jpg"])
    (spec,), _ = client.tasks.create.call_args
    assert spec.to_dict()["project_id"] == 9 and "labels" not in spec.to_dict()


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"project_id": 9, "files": ["b.jpg"]}, "one of labels or project_id"),
        ({"labels": None, "files": ["b.jpg"]}, "one of labels or project_id"),
        ({"files": ["b.jpg"], "urls": ["https://e.com/a.jpg"]}, "one of files, urls or cloud_st"),
        ({"files": ["b.jpg"], "cloud_storage_id": 1, "cloud_path": "x/"}, "one of files, urls or"),
        ({"urls": ["https://e.com/a.jpg"], "cloud_storage_id": 1, "cloud_path": "x/"}, "one of"),
        ({}, "one of files, urls or cloud_storage_id"),
        ({"cloud_storage_id": 1}, "cloud_storage_id and cloud_path together"),
        ({"cloud_path": "street/"}, "cloud_storage_id and cloud_path together"),
        ({"files": ["../x.jpg"]}, "CVAT_MCP_WORKDIR"),
        ({"files": ["/etc/passwd"]}, "CVAT_MCP_WORKDIR"),
        ({"files": ["."]}, "CVAT_MCP_WORKDIR"),
        ({"files": ["empty"]}, "no files"),
        ({"files": ["missing.jpg"]}, "no such file"),
        ({"urls": ["ftp://h/x.jpg"]}, "http"),
        ({"urls": ["file:///x"]}, "http"),
        ({"name": " ", "files": ["b.jpg"]}, "name"),
    ],
)
def test_create_refusals(tmp_path, kw, match):
    (tmp_path / "b.jpg").write_bytes(b"x")
    (tmp_path / "empty").mkdir()
    client, _ = _create_client()
    with pytest.raises(InvalidArgument, match=match):
        _create(client, tmp_path, **kw)
    client.tasks.create.assert_not_called()


@pytest.mark.parametrize("remove_error", [None, RuntimeError("remove failed")])
def test_failed_upload_removes_the_task(tmp_path, remove_error):
    client, task = _create_client(BackgroundRequestException("bad image"), remove_error)
    with pytest.raises(BackgroundRequestException, match="bad image"):
        _create(client, tmp_path, urls=["https://example.com/a.jpg"])
    task.remove.assert_called_once()


def _uploaded(task):
    _, kwargs = task.upload_data.call_args
    return kwargs["resources"], kwargs["params"]


def test_create_from_bucket_folder(tmp_path):
    client, task = _create_client()
    result = _create(client, tmp_path, cloud_storage_id=1, cloud_path="street/")
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


def test_folder_path_without_slash_is_a_folder(tmp_path):
    client, task = _create_client()
    _create(client, tmp_path, cloud_storage_id=1, cloud_path="street")
    assert _uploaded(task)[0] == ["street/"]


def test_create_from_bucket_object(tmp_path):
    client, task = _create_client()
    _create(client, tmp_path, cloud_storage_id=1, cloud_path="street/a.jpg")
    resources, params = _uploaded(task)
    assert resources == ["street/a.jpg"]
    assert "filename_pattern" not in params


def test_create_from_glob(tmp_path):
    client, task = _create_client()
    _create(client, tmp_path, cloud_storage_id=1, cloud_path="street/*.jpg")
    resources, params = _uploaded(task)
    assert resources == [] and params["filename_pattern"] == "street/*.jpg"


def test_background_failure_names_the_path(tmp_path):
    client, task = _create_client(BackgroundRequestException("No media data found"))
    with pytest.raises(
        InvalidArgument, match="No media data found.*cloud_path 'nothing/'"
    ) as error:
        _create(client, tmp_path, cloud_storage_id=1, cloud_path="nothing/")
    assert "update_cloud_storage" in str(error.value)
    task.remove.assert_called_once()


def test_unknown_storage_removes_the_task(tmp_path):
    client, task = _create_client(ApiException(status=404))
    with pytest.raises(ApiException):
        _create(client, tmp_path, cloud_storage_id=99, cloud_path="street/")
    task.remove.assert_called_once()
    client.api_client.cloudstorages_api.destroy.assert_not_called()


def test_unavailable_storage_points_at_its_keys(tmp_path):
    unavailable = ApiException(status=400)
    unavailable.body = b"The specified cloud storage 'x' is not available."
    client, task = _create_client(unavailable)
    with pytest.raises(InvalidArgument, match="update_cloud_storage"):
        _create(client, tmp_path, cloud_storage_id=1, cloud_path="street/")
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


def test_resolve_files_refuses_a_symlink_out_of_the_workdir(tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("key")
    workdir = tmp_path / "work"
    (workdir / "d").mkdir(parents=True)
    (workdir / "d" / "ok.jpg").write_bytes(b"x")
    (workdir / "d" / "link.jpg").symlink_to(outside)
    with pytest.raises(InvalidArgument, match="CVAT_MCP_WORKDIR"):
        resolve_files(workdir, ["d"])


def test_resolve_files_skips_hidden_files(tmp_path):
    (tmp_path / "d" / ".cache").mkdir(parents=True)
    for name in ("d/1.jpg", "d/.DS_Store", "d/.cache/x.jpg"):
        (tmp_path / name).write_bytes(b"x")
    assert resolve_files(tmp_path, ["d"]) == [tmp_path / "d/1.jpg"]


def test_resolve_files_refuses_duplicate_names(tmp_path):
    for name in ("d/cam1/0001.jpg", "d/cam2/0001.jpg"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"x")
    with pytest.raises(InvalidArgument, match="0001.jpg"):
        resolve_files(tmp_path, ["d"])


def test_resolve_files_limits_the_expanded_count(tmp_path, monkeypatch):
    monkeypatch.setattr("cvat_mcp.services.task_edits.MAX_FILES", 2)
    for i in range(3):
        (tmp_path / "d").mkdir(exist_ok=True)
        (tmp_path / "d" / f"{i}.jpg").write_bytes(b"x")
    with pytest.raises(InvalidArgument, match="3 files"):
        resolve_files(tmp_path, ["d"])


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
