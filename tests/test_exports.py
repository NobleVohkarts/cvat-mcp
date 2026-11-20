from pathlib import Path
from unittest.mock import Mock

import pytest
from cvat_sdk.core.proxies.types import Location
from fakes import Obj

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.exports import (
    check_format,
    default_filename,
    export_dataset,
    unique_path,
)

EXPORTERS = [
    Obj(name="COCO 1.0", enabled=True),
    Obj(name="Datumaro 1.0", enabled=True),
    Obj(name="Old 1.0", enabled=False),
]


def test_default_filename():
    assert default_filename(7, "COCO 1.0", False) == "task-7-coco-1-0.zip"
    assert default_filename(7, "YOLO 1.1", True) == "task-7-yolo-1-1-images.zip"


def test_unique_path(tmp_path):
    target = tmp_path / "a.zip"
    assert unique_path(target) == target
    target.touch()
    assert unique_path(target) == tmp_path / "a-1.zip"
    (tmp_path / "a-1.zip").touch()
    assert unique_path(target) == tmp_path / "a-2.zip"


def _client(task=None):
    server_api = Obj(retrieve_annotation_formats=lambda: (Obj(exporters=EXPORTERS), None))
    return Obj(
        api_client=Obj(server_api=server_api),
        tasks=Obj(retrieve=lambda _id: task),
    )


def test_check_format_accepts_enabled_exporter():
    check_format(_client(), "COCO 1.0")


@pytest.mark.parametrize("name", ["coco 1.0", "Old 1.0", "COCO"])
def test_check_format_lists_valid_names(name):
    with pytest.raises(InvalidArgument, match="valid: COCO 1.0, Datumaro 1.0"):
        check_format(_client(), name)


def _task():
    def write(_format, path, **_kwargs):
        Path(path).write_bytes(b"12345")

    return Obj(export_dataset=Mock(side_effect=write))


def test_export_writes_into_workdir_with_local_location(tmp_path):
    task = _task()
    result = export_dataset(_client(task), tmp_path / "work", 7, "COCO 1.0", False, None)
    assert result.path == "task-7-coco-1-0.zip"
    assert result.size_bytes == 5 and result.format == "COCO 1.0" and not result.include_images
    kwargs = task.export_dataset.call_args.kwargs
    assert kwargs["location"] == Location.LOCAL and kwargs["include_images"] is False


def test_second_export_gets_a_new_name(tmp_path):
    task = _task()
    first = export_dataset(_client(task), tmp_path, 7, "COCO 1.0", False, None)
    (tmp_path / first.path).write_bytes(b"keep")
    second = export_dataset(_client(task), tmp_path, 7, "COCO 1.0", False, None)
    assert second.path == "task-7-coco-1-0-1.zip"
    assert (tmp_path / first.path).read_bytes() == b"keep"


def test_filename_gets_zip_suffix_and_directories(tmp_path):
    result = export_dataset(_client(_task()), tmp_path, 7, "COCO 1.0", False, "sub/out")
    assert result.path == "sub/out.zip" and (tmp_path / "sub" / "out.zip").exists()


@pytest.mark.parametrize(
    "fmt, filename, match",
    [("COCO 1.0", "../x", "CVAT_MCP_WORKDIR"), ("YOLO 9", None, "unknown export format")],
)
def test_rejected_before_exporting(tmp_path, fmt, filename, match):
    task = _task()
    with pytest.raises(InvalidArgument, match=match):
        export_dataset(_client(task), tmp_path, 7, fmt, False, filename)
    task.export_dataset.assert_not_called()
