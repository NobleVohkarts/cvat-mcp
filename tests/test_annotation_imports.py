from unittest.mock import Mock

import pytest
from fakes import Obj, fake_task, gt_job, shape, tag

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.annotation_imports import import_annotations

IMPORTERS = [
    Obj(name="COCO 1.0", enabled=True, dimension="2d"),
    Obj(name="YOLO 1.1", enabled=True, dimension="2d"),
    Obj(name="Off 1.0", enabled=False, dimension="2d"),
]
SHAPES = [shape(1, 1, 0, [0, 0, 5, 5]), shape(2, 2, 2, [1, 1, 4, 4])]


def _task(shapes=tuple(SHAPES), **kw):
    return fake_task(
        shapes=shapes, tags=[tag(3, 1, 0)], dimension="2d", import_annotations=Mock(), **kw
    )


def _client(task, job=None, **api):
    formats = Obj(retrieve_annotation_formats=Mock(return_value=(Obj(importers=IMPORTERS), None)))
    return Obj(
        tasks=Obj(retrieve=Mock(return_value=task)),
        jobs=Obj(retrieve=Mock(return_value=job)),
        api_client=Obj(server_api=formats, **api),
    )


def _import(client, workdir, **kw):
    args = {
        "job_id": None,
        "task_id": None,
        "cloud_url": None,
        "file": None,
        "format_name": "COCO 1.0",
        "mode": "append",
    }
    return import_annotations(client, workdir, **(args | kw))


@pytest.mark.parametrize("target", [{}, {"job_id": 30, "task_id": 1}])
def test_needs_exactly_one_target(tmp_path, target):
    client = _client(_task())
    with pytest.raises(InvalidArgument, match="exactly one of job_id or task_id"):
        _import(client, tmp_path, file="a.json", **target)
    client.tasks.retrieve.assert_not_called()


@pytest.mark.parametrize("source", [{}, {"file": "a.json", "cloud_url": "s3://b/a.json"}])
def test_needs_exactly_one_source(tmp_path, source):
    client = _client(_task())
    with pytest.raises(InvalidArgument, match="exactly one of cloud_url or file"):
        _import(client, tmp_path, task_id=1, **source)
    client.tasks.retrieve.assert_not_called()


@pytest.mark.parametrize("name", ["Off 1.0", "Nope"])
def test_unknown_or_disabled_format_lists_importers(tmp_path, name):
    (tmp_path / "a.json").write_text("{}")
    task = _task()
    with pytest.raises(InvalidArgument, match="unknown import format.*COCO 1.0, YOLO 1.1$"):
        _import(_client(task), tmp_path, task_id=1, file="a.json", format_name=name)
    task.import_annotations.assert_not_called()


def test_local_file_imports_into_task(tmp_path):
    (tmp_path / "labels.json").write_text("{}")
    task = _task()
    result = _import(_client(task), tmp_path, task_id=1, file="labels.json")
    task.import_annotations.assert_called_once_with(
        "COCO 1.0", tmp_path / "labels.json", import_mode="append", status_check_period=1
    )
    assert (result.task_id, result.job_id, result.source) == (1, None, "labels.json")
    assert (result.format, result.mode) == ("COCO 1.0", "append")
    assert (result.shapes, result.tracks, result.tags) == (2, 0, 1)
    assert result.cloud_storage_id is None and result.cloud_storage_registered is False
    assert result.note is None


def test_local_file_imports_into_job_with_replace(tmp_path):
    (tmp_path / "gt.json").write_text("{}")
    job = gt_job(shapes=[SHAPES[0]], import_annotations=Mock())
    task = _task()
    result = _import(_client(task, job), tmp_path, job_id=30, file="gt.json", mode="replace")
    job.import_annotations.assert_called_once_with(
        "COCO 1.0", tmp_path / "gt.json", import_mode="replace", status_check_period=1
    )
    task.import_annotations.assert_not_called()
    assert (result.task_id, result.job_id, result.mode) == (1, 30, "replace")
    assert (result.shapes, result.tracks, result.tags) == (1, 0, 0)  # the job's, not the task's


@pytest.mark.parametrize(
    "name, match",
    [
        ("missing.json", "no such file in the work directory: 'missing.json'"),
        ("../x.json", "inside CVAT_MCP_WORKDIR"),
        ("folder", "'folder' is not a file"),
    ],
)
def test_local_file_must_exist_in_workdir(tmp_path, name, match):
    (tmp_path / "folder").mkdir()
    task = _task()
    with pytest.raises(InvalidArgument, match=match):
        _import(_client(task), tmp_path, task_id=1, file=name)
    task.import_annotations.assert_not_called()


def test_zero_objects_adds_note(tmp_path):
    (tmp_path / "a.json").write_text("{}")
    task = fake_task(dimension="2d", import_annotations=Mock())
    result = _import(_client(task), tmp_path, task_id=1, file="a.json")
    assert (result.shapes, result.tracks, result.tags) == (0, 0, 0)
    assert "file names" in result.note and "frame names" in result.note
