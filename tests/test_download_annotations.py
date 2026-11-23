import json
from unittest.mock import Mock

import pytest
from fakes import Obj, annotation_job, fake_client, fake_task, gt_job, job_client, shape, tag

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.annotations import JobFrame, download_annotations
from cvat_mcp.tools import annotations as tools


@pytest.mark.parametrize("task_id, job_id", [(None, None), (1, 5)])
def test_needs_exactly_one_target(task_id, job_id):
    client = Mock()
    with pytest.raises(InvalidArgument, match="exactly one"):
        download_annotations(client, task_id, job_id)
    assert client.mock_calls == []


def _one_job(shapes=(), tags=(), tracks=(), **task_kw):
    task = fake_task(size=10, **task_kw)
    return download_annotations(
        job_client(task, annotation_job(5, 0, 9, shapes, tags, tracks)), None, 5
    )


def test_one_job_shape_keys_and_values():
    polygon = shape(
        1, 1, 3, [1, 2, 3, 4], type="polygon", attributes=[Obj(spec_id=11, value="red")]
    )
    [result] = _one_job([polygon])
    assert result.model_dump(exclude_none=True)["shapes"][0] == {
        "id": 1,
        "frame": 3,
        "label": "car",
        "type": "polygon",
        "points": [1, 2, 3, 4],
        "group": 0,
        "occluded": False,
        "z_order": 0,
        "source": "manual",
        "attributes": {"color": "red"},
    }
    assert (result.task_id, result.job_id, result.job_type) == (1, 5, "annotation")
    assert result.frames[0] == JobFrame(frame=0, name="frame_000000.jpg", width=640, height=480)
    assert len(result.frames) == 10


def test_json_key_order_matches_the_spec():
    [result] = _one_job([shape(1, 1, 0, [1, 2, 3, 4])])
    data = result.model_dump(exclude_none=True)
    assert list(data) == [
        "task_id", "job_id", "job_type", "frames", "shapes", "tags", "skipped_tracks"
    ]  # fmt: skip
    assert list(data["shapes"][0]) == [
        "id", "frame", "label", "type", "points", "group", "occluded", "z_order", "source"
    ]  # fmt: skip  # no "attributes": left out when empty


def test_tags_and_tracks():
    [result] = _one_job(tags=[tag(7, 2, 0)], tracks=[Obj(shapes=[])])
    assert result.tags[0].model_dump(exclude_none=True) == {
        "id": 7,
        "frame": 0,
        "label": "person",
        "source": "manual",
    }
    assert result.skipped_tracks == 1


def test_task_id_returns_annotation_jobs_in_order():
    jobs = [annotation_job(9, 5, 9), annotation_job(8, 0, 4), gt_job(job_id=30)]
    result = download_annotations(fake_client(fake_task(size=10, jobs=jobs)), 1, None)
    assert [r.job_id for r in result] == [8, 9]
    assert [f.frame for f in result[0].frames] == [0, 1, 2, 3, 4]


def test_ground_truth_by_job_id():
    [result] = download_annotations(job_client(fake_task(), gt_job(included=(0, 2))), None, 30)
    assert result.job_type == "ground_truth"
    assert [f.frame for f in result.frames] == [0, 2]


def test_deleted_frames_dropped():
    shapes = [shape(1, 1, 3, [1, 2, 3, 4]), shape(2, 1, 4, [1, 2, 3, 4])]
    [result] = _one_job(shapes, tags=[tag(7, 2, 3)], deleted_frames=(3,))
    assert 3 not in [f.frame for f in result.frames]
    assert [s.id for s in result.shapes] == [2]
    assert result.tags == []


def test_video_task_single_meta_entry():
    [result] = _one_job(meta_frames=[Obj(width=1920, height=1080, name="clip.mp4")])
    assert result.frames == [
        JobFrame(frame=f, name="clip.mp4", width=1920, height=1080) for f in range(10)
    ]


def _shapes(n, frame):
    return [shape(i, 1, frame, [1, 2, 3, 4]) for i in range(n)]


def test_cap_counts_all_jobs():
    jobs = [annotation_job(8, 0, 4, _shapes(1200, 0)), annotation_job(9, 5, 9, _shapes(1200, 5))]
    with pytest.raises(InvalidArgument) as error:
        download_annotations(fake_client(fake_task(size=10, jobs=jobs)), 1, None)
    for part in ("2400", "2000", "job 8: 1200", "job 9: 1200", "job_id"):
        assert part in str(error.value)


def test_exactly_cap_is_fine():
    [result] = _one_job(_shapes(2000, 0))
    assert len(result.shapes) == 2000


def test_task_without_annotation_jobs():
    with pytest.raises(InvalidArgument, match="task 1 has no annotation jobs"):
        download_annotations(fake_client(fake_task(jobs=[gt_job()])), 1, None)


def test_job_id_reads_the_jobs_task():
    task = fake_task(task_id=7)
    retrieve = Mock(return_value=task)
    client = Obj(
        tasks=Obj(retrieve=retrieve),
        jobs=Obj(retrieve=lambda _id: annotation_job(5, 0, 3, task_id=7)),
    )
    [result] = download_annotations(client, None, 5)
    retrieve.assert_called_once_with(7)
    assert result.task_id == 7


def test_tool_returns_json_resources():
    jobs = [annotation_job(8, 0, 4, [shape(1, 1, 0, [1, 2, 3, 4])]), annotation_job(9, 5, 9)]
    ctx = Obj(request_context=Obj(lifespan_context=fake_client(fake_task(size=10, jobs=jobs))))
    blocks = tools.download_annotations(ctx, task_id=1)
    assert [b.type for b in blocks] == ["resource", "resource"]
    for block, job_id in zip(blocks, [8, 9]):
        resource = block.resource
        assert resource.mime_type == "application/json"
        assert str(resource.uri) == f"cvat://jobs/{job_id}/annotations.json"
        assert json.loads(resource.text)["job_id"] == job_id
        assert '": ' not in resource.text  # compact
