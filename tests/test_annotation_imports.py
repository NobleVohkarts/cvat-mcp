from unittest.mock import Mock

import pytest
from cvat_sdk.api_client.exceptions import ApiException
from cvat_sdk.core.exceptions import BackgroundRequestException
from fakes import Obj, cloud_storage, fake_cloudstorages_api, fake_task, gt_job, shape, tag

from cvat_mcp.errors import InvalidArgument, explain
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
        "cloud_storage_id": None,
        "cloud_path": None,
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


@pytest.mark.parametrize(
    "source", [{}, {"file": "a.json", "cloud_storage_id": 1, "cloud_path": "a.json"}]
)
def test_needs_exactly_one_source(tmp_path, source):
    client = _client(_task())
    with pytest.raises(InvalidArgument, match="exactly one of cloud_storage_id"):
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
    assert result.cloud_storage_id is None
    assert result.note is None


def test_local_file_imports_into_job_with_replace(tmp_path):
    (tmp_path / "gt.json").write_text("{}")
    job = gt_job(shapes=[SHAPES[0]], dimension="2d", import_annotations=Mock())
    task = _task()
    client = _client(task, job)
    result = _import(client, tmp_path, job_id=30, file="gt.json", mode="replace")
    client.tasks.retrieve.assert_not_called()  # the job knows its task and dimension
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
        ("folder", "no such file in the work directory: 'folder'"),
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


STARTED = (None, Obj(data=b'{"rq_id": "rq-1"}'))


def _cloud_client(task, job=None, wait_error=None, storages=(cloud_storage(1, "cvat"),)):
    client = _client(
        task,
        job,
        cloudstorages_api=fake_cloudstorages_api(storages),
        jobs_api=Obj(create_annotations=Mock(return_value=STARTED)),
        tasks_api=Obj(create_annotations=Mock(return_value=STARTED)),
    )
    client.wait_for_completion = Mock(side_effect=wait_error)
    return client


def test_cloud_import_into_gt_job_uses_storage(tmp_path):
    job = gt_job(shapes=[SHAPES[0]], dimension="2d")
    client = _cloud_client(_task(), job)
    result = _import(
        client, tmp_path, job_id=30, cloud_storage_id=1, cloud_path="annotations/street.json"
    )
    client.api_client.jobs_api.create_annotations.assert_called_once_with(
        30,
        format="COCO 1.0",
        location="cloud_storage",
        cloud_storage_id=1,
        filename="annotations/street.json",
        import_mode="append",
        _parse_response=False,
    )
    client.wait_for_completion.assert_called_once_with("rq-1", status_check_period=1)
    client.api_client.tasks_api.create_annotations.assert_not_called()
    assert result.cloud_storage_id == 1
    assert (result.job_id, result.shapes) == (30, 1)
    assert result.source == "annotations/street.json"


def test_cloud_import_into_task_uses_tasks_api(tmp_path):
    client = _cloud_client(_task())
    result = _import(
        client, tmp_path, task_id=1, cloud_storage_id=1, cloud_path="a.json", mode="replace"
    )
    (task_id,), kwargs = client.api_client.tasks_api.create_annotations.call_args
    assert (task_id, kwargs["filename"], kwargs["import_mode"]) == (1, "a.json", "replace")
    client.api_client.jobs_api.create_annotations.assert_not_called()
    assert result.shapes == 2


@pytest.mark.parametrize("path", ["street/", "street", "*.json", ""])
def test_cloud_path_must_name_one_object(tmp_path, path):
    client = _cloud_client(_task())
    with pytest.raises(InvalidArgument, match="cloud_path must name one annotation file"):
        _import(client, tmp_path, task_id=1, cloud_storage_id=1, cloud_path=path)
    client.api_client.tasks_api.create_annotations.assert_not_called()
    client.tasks.retrieve.assert_not_called()  # checked before any request
    client.jobs.retrieve.assert_not_called()


@pytest.mark.parametrize("half", [{"cloud_storage_id": 1}, {"cloud_path": "a.json"}])
def test_cloud_args_come_together(tmp_path, half):
    client = _cloud_client(_task())
    with pytest.raises(InvalidArgument, match="cloud_storage_id and cloud_path together"):
        _import(client, tmp_path, task_id=1, **half)
    client.tasks.retrieve.assert_not_called()


def test_organization_storage_refused_before_the_import(tmp_path):
    client = _cloud_client(_task(), storages=[cloud_storage(1, "cvat", org=5)])
    with pytest.raises(InvalidArgument, match="cloud storage #1 belongs to an organization"):
        _import(client, tmp_path, task_id=1, cloud_storage_id=1, cloud_path="a.json")
    client.api_client.tasks_api.create_annotations.assert_not_called()


def test_file_outside_the_storage_prefix_refused(tmp_path):
    client = _cloud_client(_task(), storages=[cloud_storage(1, "cvat", "prefix=street/")])
    with pytest.raises(InvalidArgument, match="covers only 'street/'"):
        _import(client, tmp_path, task_id=1, cloud_storage_id=1, cloud_path="labels.json")
    client.api_client.tasks_api.create_annotations.assert_not_called()
    _import(client, tmp_path, task_id=1, cloud_storage_id=1, cloud_path="street/labels.json")
    client.api_client.tasks_api.create_annotations.assert_called_once()


def test_unknown_storage_refused_before_the_import(tmp_path):
    client = _cloud_client(_task())
    client.api_client.cloudstorages_api.retrieve = Mock(side_effect=ApiException(status=404))
    with pytest.raises(InvalidArgument, match="no cloud storage #9.*list_cloud_storages"):
        _import(client, tmp_path, task_id=1, cloud_storage_id=9, cloud_path="a.json")
    client.api_client.tasks_api.create_annotations.assert_not_called()


NO_MATCH = BackgroundRequestException("Could not match item id: 'x' with any task frame")


def test_failed_import_points_at_storage_keys(tmp_path):
    client = _cloud_client(_task(), wait_error=NO_MATCH)
    with pytest.raises(
        InvalidArgument, match="Could not match item id.*file names.*cloud storage #1.*update_cloud"
    ):
        _import(client, tmp_path, task_id=1, cloud_storage_id=1, cloud_path="labels.json")


def test_local_failure_is_explained_too(tmp_path):
    (tmp_path / "a.json").write_text("{}")
    task = _task()
    task.import_annotations.side_effect = NO_MATCH
    with pytest.raises(InvalidArgument, match="Could not match item id.*file names") as error:
        _import(_client(task), tmp_path, task_id=1, file="a.json")
    assert "cloud storage" not in str(error.value)


def test_import_already_running_is_explained(tmp_path):
    client = _cloud_client(_task(), gt_job(dimension="2d"))
    client.api_client.jobs_api.create_annotations.side_effect = ApiException(409, "Conflict")
    with pytest.raises(InvalidArgument, match="into job 30 is still running") as error:
        _import(client, tmp_path, job_id=30, cloud_storage_id=1, cloud_path="labels.json")
    assert "duplicate" in str(error.value) and "check" in str(error.value)


def test_other_409s_keep_the_generic_message():
    # e.g. CloudStorageMissingError when reading frames: waiting would not help
    assert explain(ApiException(409, "Conflict")).startswith("CVAT answered 409 Conflict")
