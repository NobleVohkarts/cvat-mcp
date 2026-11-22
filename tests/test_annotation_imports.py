from unittest.mock import Mock

import pytest
from cvat_sdk.core.exceptions import BackgroundRequestException
from fakes import Obj, cloud_storage, fake_cloudstorages_api, fake_task, gt_job, shape, tag

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


R2 = "https://abc.r2.cloudflarestorage.com"
STORAGE_1 = cloud_storage(1, "cvat", f"endpoint_url={R2}&region=auto")
STARTED = (None, Obj(data=b'{"rq_id": "rq-1"}'))


def _cloud_client(task, job=None, storages=(STORAGE_1,), wait_error=None):
    storages_api = fake_cloudstorages_api(storages)
    storages_api.create.return_value = (cloud_storage(7, "open-bucket"), None)
    client = _client(
        task,
        job,
        cloudstorages_api=storages_api,
        jobs_api=Obj(create_annotations=Mock(return_value=STARTED)),
        tasks_api=Obj(create_annotations=Mock(return_value=STARTED)),
    )
    client.users = Obj(retrieve_current_user=lambda: Obj(id=1))
    client.wait_for_completion = Mock(side_effect=wait_error)
    return client


def test_cloud_import_into_gt_job_uses_storage(tmp_path):
    job = gt_job(shapes=[SHAPES[0]])
    client = _cloud_client(_task(), job)
    result = _import(client, tmp_path, job_id=30, cloud_url=f"{R2}/cvat/annotations/street.json")
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
    client.api_client.cloudstorages_api.create.assert_not_called()
    assert (result.cloud_storage_id, result.cloud_storage_registered) == (1, False)
    assert (result.job_id, result.shapes) == (30, 1)
    assert result.source == f"{R2}/cvat/annotations/street.json"


def test_cloud_import_into_task_uses_tasks_api(tmp_path):
    client = _cloud_client(_task())
    result = _import(client, tmp_path, task_id=1, cloud_url=f"{R2}/cvat/a.json", mode="replace")
    (task_id,), kwargs = client.api_client.tasks_api.create_annotations.call_args
    assert (task_id, kwargs["filename"], kwargs["import_mode"]) == (1, "a.json", "replace")
    client.api_client.jobs_api.create_annotations.assert_not_called()
    assert result.shapes == 2


@pytest.mark.parametrize(
    "url", [f"{R2}/cvat/street/", f"{R2}/cvat/street", f"{R2}/cvat/*.json", f"{R2}/cvat"]
)
def test_cloud_url_must_name_one_object(tmp_path, url):
    client = _cloud_client(_task())
    with pytest.raises(InvalidArgument, match="one annotation file"):
        _import(client, tmp_path, task_id=1, cloud_url=url)
    client.api_client.tasks_api.create_annotations.assert_not_called()
    client.api_client.cloudstorages_api.list.assert_not_called()


def test_public_bucket_is_registered_without_keys(tmp_path):
    client = _cloud_client(_task(), storages=())
    result = _import(client, tmp_path, task_id=1, cloud_url="s3://open-bucket/labels.json")
    assert (result.cloud_storage_id, result.cloud_storage_registered) == (7, True)
    client.api_client.cloudstorages_api.destroy.assert_not_called()


NO_MATCH = BackgroundRequestException("Could not match item id: 'x' with any task frame")


def test_failed_import_explains_and_forgets_new_storage(tmp_path):
    client = _cloud_client(_task(), storages=(), wait_error=NO_MATCH)
    with pytest.raises(InvalidArgument, match="Could not match item id.*file names") as error:
        _import(client, tmp_path, task_id=1, cloud_url="s3://open-bucket/labels.json")
    client.api_client.cloudstorages_api.destroy.assert_called_once_with(7)
    assert "update_cloud_storage" not in str(error.value)


def test_failed_import_keeps_found_storage(tmp_path):
    client = _cloud_client(_task(), wait_error=NO_MATCH)
    with pytest.raises(InvalidArgument, match="cloud storage #1.*update_cloud_storage"):
        _import(client, tmp_path, task_id=1, cloud_url=f"{R2}/cvat/labels.json")
    client.api_client.cloudstorages_api.destroy.assert_not_called()


def test_local_failure_is_explained_too(tmp_path):
    (tmp_path / "a.json").write_text("{}")
    task = _task()
    task.import_annotations.side_effect = NO_MATCH
    with pytest.raises(InvalidArgument, match="Could not match item id.*file names") as error:
        _import(_client(task), tmp_path, task_id=1, file="a.json")
    assert "cloud storage" not in str(error.value)


def test_unexpected_error_passes_through_and_forgets_new_storage(tmp_path):
    client = _cloud_client(_task(), storages=(), wait_error=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        _import(client, tmp_path, task_id=1, cloud_url="s3://open-bucket/labels.json")
    client.api_client.cloudstorages_api.destroy.assert_called_once_with(7)
