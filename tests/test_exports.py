from pathlib import Path
from unittest.mock import Mock

import pytest
from cvat_sdk.core.proxies.types import Location
from fakes import Obj, fake_client

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.exports import (
    check_format,
    default_filename,
    export_dataset,
    reserve_path,
)

EXPORTERS = [
    Obj(name="COCO 1.0", enabled=True, dimension="2d"),
    Obj(name="Datumaro 1.0", enabled=True, dimension="2d"),
    Obj(name="Old 1.0", enabled=False, dimension="2d"),
    Obj(name="Kitti Raw Format 1.0", enabled=True, dimension="3d"),
]
FORMATS = Obj(server_api=Obj(retrieve_annotation_formats=lambda: (Obj(exporters=EXPORTERS), None)))


def test_default_filename_marks_images():
    assert default_filename(7, "YOLO 1.1", True) == "task-7-yolo-1-1-images.zip"


def test_reserve_path_skips_existing_files(tmp_path):
    (tmp_path / "a.zip").touch()
    (tmp_path / "a-1.zip").touch()
    with reserve_path(tmp_path / "a.zip") as free:
        assert free == tmp_path / "a-2.zip"


@pytest.mark.parametrize("name", ["coco 1.0", "Old 1.0", "COCO", "Kitti Raw Format 1.0"])
def test_check_format_lists_valid_names_for_the_dimension(name):
    with pytest.raises(InvalidArgument, match=r"valid for a 2d task: COCO 1.0, Datumaro 1.0$"):
        check_format(Obj(api_client=FORMATS), name, "2d")


def test_reserved_names_differ_while_exports_run(tmp_path):
    with reserve_path(tmp_path / "a.zip") as first, reserve_path(tmp_path / "a.zip") as second:
        assert first != second
    with reserve_path(tmp_path / "a.zip") as again:
        assert again == tmp_path / "a.zip"  # released after the export


def _task():
    def write(_format, path, **_kwargs):
        Path(path).write_bytes(b"12345")
        return Path(path)  # the SDK returns where it saved the file

    return Obj(export_dataset=Mock(side_effect=write), dimension="2d")


def _export(tmp_path, task, fmt="COCO 1.0", filename=None):
    return export_dataset(fake_client(task, api_client=FORMATS), tmp_path, 7, fmt, False, filename)


def test_export_downloads_locally_and_polls_often(tmp_path):
    task = _task()
    result = _export(tmp_path, task)
    assert result.path == str(tmp_path / "task-7-coco-1-0.zip") and result.size_bytes == 5
    kwargs = task.export_dataset.call_args.kwargs
    assert kwargs["location"] == Location.LOCAL and kwargs["include_images"] is False
    assert kwargs["status_check_period"] == 1


def test_second_export_never_overwrites(tmp_path):
    task = _task()
    first = Path(_export(tmp_path, task).path)
    first.write_bytes(b"keep")
    assert Path(_export(tmp_path, task).path).name == "task-7-coco-1-0-1.zip"
    assert first.read_bytes() == b"keep"


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("sub/out", "sub/out.zip"),
        ("task7-COCO-1.0", "task7-COCO-1.0.zip"),
        ("out.ZIP", "out.ZIP"),
        ("archive.tar.gz", "archive.tar.gz.zip"),
    ],
)
def test_filename_always_ends_in_zip(tmp_path, filename, expected):
    assert _export(tmp_path, _task(), filename=filename).path == str(tmp_path / expected)


@pytest.mark.parametrize(
    "fmt, filename, match",
    [("COCO 1.0", "../x", "CVAT_MCP_WORKDIR"), ("YOLO 9", None, "unknown export format")],
)
def test_rejected_before_exporting(tmp_path, fmt, filename, match):
    task = _task()
    with pytest.raises(InvalidArgument, match=match):
        _export(tmp_path, task, fmt, filename)
    task.export_dataset.assert_not_called()
