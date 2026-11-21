from unittest.mock import Mock

import pytest
from cvat_sdk.core.exceptions import BackgroundRequestException
from cvat_sdk.core.proxies.tasks import ResourceType
from fakes import Obj

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.task_edits import (
    AttributeSpec,
    LabelSpec,
    create_task,
    label_requests,
)

COLOR = AttributeSpec(name="color", input_type="select", values=["red", "blue"])


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
        (
            [LabelSpec(name="car", attributes=[AttributeSpec(name="c", input_type="select")])],
            (),
            "'c'",
        ),
        (
            [
                LabelSpec(
                    name="car",
                    attributes=[AttributeSpec(name="n", input_type="number", values=["0", "10"])],
                )
            ],
            (),
            "'n'",
        ),
        (
            [
                LabelSpec(
                    name="car",
                    attributes=[
                        AttributeSpec(name="n", input_type="number", values=["0", "10", "3"])
                    ],
                )
            ],
            (),
            "'n'",
        ),
        (
            [
                LabelSpec(
                    name="car",
                    attributes=[AttributeSpec(name="p", input_type="checkbox", values=["yes"])],
                )
            ],
            (),
            "'p'",
        ),
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
    task.fetch = Mock()
    return Obj(tasks=Obj(create=Mock(return_value=task))), task


def _create(client, workdir, **kw):
    args = {
        "name": "Scratch",
        "labels": [LabelSpec(name="car")],
        "project_id": None,
        "files": None,
        "urls": None,
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
        ({"files": ["b.jpg"], "urls": ["https://e.com/a.jpg"]}, "one of files or urls"),
        ({}, "one of files or urls"),
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
