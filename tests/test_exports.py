from unittest.mock import Mock

import pytest
from cvat_sdk.api_client.exceptions import ApiException
from cvat_sdk.core.exceptions import BackgroundRequestException
from cvat_sdk.core.proxies.types import Location
from fakes import Obj, cloud_storage, content_listing, fake_client, fake_cloudstorages_api

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.exports import check_format, export_dataset

EXPORTERS = [
    Obj(name="COCO 1.0", enabled=True, dimension="2d"),
    Obj(name="Datumaro 1.0", enabled=True, dimension="2d"),
    Obj(name="Old 1.0", enabled=False, dimension="2d"),
    Obj(name="Kitti Raw Format 1.0", enabled=True, dimension="3d"),
]
FORMATS = Obj(server_api=Obj(retrieve_annotation_formats=lambda: (Obj(exporters=EXPORTERS), None)))


@pytest.mark.parametrize("name", ["coco 1.0", "Old 1.0", "COCO", "Kitti Raw Format 1.0"])
def test_check_format_lists_valid_names_for_the_dimension(name):
    with pytest.raises(InvalidArgument, match=r"valid for a 2d task: COCO 1.0, Datumaro 1.0$"):
        check_format(Obj(api_client=FORMATS), name, "2d")


def test_check_format_import_direction_reads_importers():
    importers = [Obj(name="YOLO 1.1", enabled=True, dimension="2d")]
    formats = Obj(
        retrieve_annotation_formats=lambda: (Obj(exporters=EXPORTERS, importers=importers), None)
    )
    client = Obj(api_client=Obj(server_api=formats))
    check_format(client, "YOLO 1.1", "2d", "import")
    with pytest.raises(InvalidArgument, match=r"unknown import format 'COCO 1.0'.*: YOLO 1.1$"):
        check_format(client, "COCO 1.0", "2d", "import")


def _client(content=(), storage=None, upload_error=None):
    """Task 7 (2d) and cloud storage #1 whose listing holds `content` ((name, type) pairs)."""
    task = Obj(
        id=7, dimension="2d", export_dataset=Mock(return_value=None, side_effect=upload_error)
    )
    api = fake_cloudstorages_api([storage or cloud_storage(1, "cvat")])
    api.retrieve_content_v2 = listing = content_listing(list(content))
    client = fake_client(task, api_client=Obj(server_api=FORMATS.server_api, cloudstorages_api=api))
    return client, task, listing


def _export(client, cloud_path="exports/a.zip", overwrite=False, fmt="COCO 1.0"):
    return export_dataset(client, 7, fmt, 1, cloud_path, False, overwrite)


def test_export_sends_the_key_to_the_storage():
    client, task, _ = _client()
    result = _export(client)
    assert (result.cloud_storage_id, result.cloud_path, result.replaced) == (
        1,
        "exports/a.zip",
        False,
    )
    task.export_dataset.assert_called_once_with(
        "COCO 1.0",
        "exports/a.zip",
        include_images=False,
        location=Location.CLOUD_STORAGE,
        cloud_storage_id=1,
        status_check_period=1,  # the SDK default (5 s) makes even a tiny export take 5 s
    )


@pytest.mark.parametrize(
    "cloud_path, key",
    [
        (" /exports/a.zip ", "exports/a.zip"),
        ("exports/a", "exports/a.zip"),
        ("a.ZIP", "a.ZIP"),
        ("a.tar.gz", "a.tar.gz.zip"),
    ],
)
def test_key_is_normalized_and_gets_zip(cloud_path, key):
    client, task, listing = _client()
    assert _export(client, cloud_path).cloud_path == key
    assert listing.call_args.kwargs["prefix"] == key
    assert task.export_dataset.call_args.args[1] == key


@pytest.mark.parametrize("cloud_path", ["", " / ", "exports/", "a*.zip", "a?.zip"])
def test_bad_cloud_path_refused_before_any_request(cloud_path):
    client, task, listing = _client()
    with pytest.raises(InvalidArgument, match="cloud_path must name the file to write"):
        _export(client, cloud_path)
    listing.assert_not_called()
    task.export_dataset.assert_not_called()


def test_existing_object_refused_without_overwrite():
    client, task, _ = _client([("a.zip", "REG")])
    with pytest.raises(
        InvalidArgument,
        match=r"cloud storage #1 already has exports/a.zip; ask the user whether to replace it.*"
        r"overwrite=true",
    ):
        _export(client)
    task.export_dataset.assert_not_called()


def test_overwrite_replaces_and_says_so():
    client, task, _ = _client([("a.zip", "REG")])
    assert _export(client, overwrite=True).replaced is True
    task.export_dataset.assert_called_once()


def test_unknown_format_refused_first():
    client, task, listing = _client()
    with pytest.raises(InvalidArgument, match="unknown export format"):
        _export(client, fmt="YOLO 9")
    listing.assert_not_called()
    task.export_dataset.assert_not_called()


def _missing(*_args, **_kwargs):
    raise ApiException(status=404)


@pytest.mark.parametrize(
    "storage, match",
    [
        (cloud_storage(1, "cvat", org=3), "belongs to an organization"),
        (cloud_storage(1, "cvat", "prefix=street/"), "covers only 'street/'"),
    ],
)
def test_storage_checks_apply(storage, match):
    client, task, listing = _client(storage=storage)
    with pytest.raises(InvalidArgument, match=match):
        _export(client)
    listing.assert_not_called()
    task.export_dataset.assert_not_called()


def test_unknown_storage_refused():
    client, task, _ = _client()
    client.api_client.cloudstorages_api.retrieve = _missing
    with pytest.raises(InvalidArgument, match="no cloud storage #1"):
        _export(client)
    task.export_dataset.assert_not_called()


def test_failed_upload_points_at_the_storage_keys():
    client, _, _ = _client(upload_error=BackgroundRequestException("Access Denied"))
    with pytest.raises(
        InvalidArgument, match=r"Access Denied.*cloud storage #1 can write to the bucket.*update_cl"
    ):
        _export(client)
