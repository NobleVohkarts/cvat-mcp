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
    return fake_task(shapes=shapes, tags=[tag(3, 1, 0)], dimension="2d", **kw)


STARTED = (None, Obj(data=b'{"rq_id": "rq-1"}'))


def _client(task, job=None, wait_error=None, storages=(cloud_storage(1, "cvat"),)):
    formats = Obj(retrieve_annotation_formats=Mock(return_value=(Obj(importers=IMPORTERS), None)))
    client = Obj(
        tasks=Obj(retrieve=Mock(return_value=task)),
        jobs=Obj(retrieve=Mock(return_value=job)),
        api_client=Obj(
            server_api=formats,
            cloudstorages_api=fake_cloudstorages_api(storages),
            jobs_api=Obj(create_annotations=Mock(return_value=STARTED)),
            tasks_api=Obj(create_annotations=Mock(return_value=STARTED)),
        ),
    )
    client.wait_for_completion = Mock(side_effect=wait_error)
    return client


def _import(client, **kw):
    args = {
        "job_id": None,
        "task_id": None,
        "cloud_storage_id": 1,
        "cloud_path": "a.json",
        "format_name": "COCO 1.0",
        "mode": "append",
    }
    return import_annotations(client, **(args | kw))


@pytest.mark.parametrize("target", [{}, {"job_id": 30, "task_id": 1}])
def test_needs_exactly_one_target(target):
    client = _client(_task())
    with pytest.raises(InvalidArgument, match="exactly one of job_id or task_id"):
        _import(client, **target)
    client.tasks.retrieve.assert_not_called()


@pytest.mark.parametrize("name", ["Off 1.0", "Nope"])
def test_unknown_or_disabled_format_lists_importers(name):
    client = _client(_task())
    with pytest.raises(InvalidArgument, match="unknown import format.*COCO 1.0, YOLO 1.1$"):
        _import(client, task_id=1, format_name=name)
    client.api_client.tasks_api.create_annotations.assert_not_called()


def test_zero_objects_adds_note():
    result = _import(_client(fake_task(dimension="2d")), task_id=1)
    assert (result.shapes, result.tracks, result.tags) == (0, 0, 0)
    assert "file names" in result.note and "frame names" in result.note


def test_cloud_import_into_gt_job_uses_storage():
    job = gt_job(shapes=[SHAPES[0]], dimension="2d")
    client = _client(_task(), job)
    result = _import(client, job_id=30, cloud_storage_id=1, cloud_path="annotations/street.json")
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
    assert result.cloud_path == "annotations/street.json"


def test_cloud_import_into_task_uses_tasks_api():
    client = _client(_task())
    result = _import(client, task_id=1, mode="replace")
    (task_id,), kwargs = client.api_client.tasks_api.create_annotations.call_args
    assert (task_id, kwargs["filename"], kwargs["import_mode"]) == (1, "a.json", "replace")
    client.api_client.jobs_api.create_annotations.assert_not_called()
    assert result.shapes == 2


@pytest.mark.parametrize("path", ["street/", "street", "*.json", ""])
def test_cloud_path_must_name_one_object(path):
    client = _client(_task())
    with pytest.raises(InvalidArgument, match="cloud_path must name one annotation file"):
        _import(client, task_id=1, cloud_storage_id=1, cloud_path=path)
    client.api_client.tasks_api.create_annotations.assert_not_called()
    client.tasks.retrieve.assert_not_called()  # checked before any request
    client.jobs.retrieve.assert_not_called()


def test_organization_storage_refused_before_the_import():
    client = _client(_task(), storages=[cloud_storage(1, "cvat", org=5)])
    with pytest.raises(InvalidArgument, match="cloud storage #1 belongs to an organization"):
        _import(client, task_id=1)
    client.api_client.tasks_api.create_annotations.assert_not_called()


def test_file_outside_the_storage_prefix_refused():
    client = _client(_task(), storages=[cloud_storage(1, "cvat", "prefix=street/")])
    with pytest.raises(InvalidArgument, match="covers only 'street/'"):
        _import(client, task_id=1, cloud_storage_id=1, cloud_path="labels.json")
    client.api_client.tasks_api.create_annotations.assert_not_called()
    _import(client, task_id=1, cloud_storage_id=1, cloud_path="street/labels.json")
    client.api_client.tasks_api.create_annotations.assert_called_once()


def test_unknown_storage_refused_before_the_import():
    client = _client(_task())
    client.api_client.cloudstorages_api.retrieve = Mock(side_effect=ApiException(status=404))
    with pytest.raises(InvalidArgument, match="no cloud storage #9.*list_cloud_storages"):
        _import(client, task_id=1, cloud_storage_id=9, cloud_path="a.json")
    client.api_client.tasks_api.create_annotations.assert_not_called()


NO_MATCH = BackgroundRequestException("Could not match item id: 'x' with any task frame")


def test_failed_import_points_at_storage_keys():
    client = _client(_task(), wait_error=NO_MATCH)
    with pytest.raises(
        InvalidArgument, match="Could not match item id.*file names.*cloud storage #1.*update_cloud"
    ):
        _import(client, task_id=1, cloud_storage_id=1, cloud_path="labels.json")


def test_import_already_running_is_explained():
    client = _client(_task(), gt_job(dimension="2d"))
    client.api_client.jobs_api.create_annotations.side_effect = ApiException(409, "Conflict")
    with pytest.raises(InvalidArgument, match="into job 30 is still running") as error:
        _import(client, job_id=30, cloud_storage_id=1, cloud_path="labels.json")
    assert "duplicate" in str(error.value) and "check" in str(error.value)


def test_other_409s_keep_the_generic_message():
    # e.g. CloudStorageMissingError when reading frames: waiting would not help
    assert explain(ApiException(409, "Conflict")).startswith("CVAT answered 409 Conflict")


def test_result_shows_the_key_used():
    result = _import(_client(_task()), task_id=1, cloud_path=" /annotations/a.json ")
    assert result.cloud_path == "annotations/a.json"
