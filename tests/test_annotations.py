import pytest
from fakes import Obj, fake_client, fake_task, shape

from cvat_mcp.services.annotations import check_frame, get_frame_annotations, shape_bbox


def test_shape_bbox_rectangle():
    assert shape_bbox(shape(1, 1, 0, [10, 20, 110, 220])) == [10, 20, 110, 220]


def test_shape_bbox_polygon():
    assert shape_bbox(shape(1, 1, 0, [5, 50, 30, 10, 60, 40], type="polygon")) == [5, 10, 60, 50]


def test_shape_bbox_mask_uses_trailing_box():
    rle = [500, 20, 480, 20, 3000]
    assert shape_bbox(shape(1, 1, 0, rle + [10, 20, 59, 89], type="mask")) == [10, 20, 59, 89]


def test_shape_bbox_skeleton_uses_elements():
    elements = [Obj(points=[10, 50]), Obj(points=[40, 20]), Obj(points=[30, 90])]
    skeleton = shape(1, 1, 0, [], type="skeleton", elements=elements)
    assert shape_bbox(skeleton) == [10, 20, 40, 90]


@pytest.mark.parametrize("frame", [-1, 40])
def test_check_frame_outside(frame):
    with pytest.raises(ValueError, match=r"0\.\.39"):
        check_frame(frame, 40)


def test_check_frame_inside():
    check_frame(39, 40)


def test_only_objects_of_that_frame():
    task = fake_task(
        [
            shape(1, 1, 3, [10, 20, 110, 220], attributes=[Obj(spec_id=10, value="true")]),
            shape(2, 2, 3, [0, 0, 5, 5], source="auto", occluded=True),
            shape(3, 1, 4, [1, 1, 2, 2]),
        ]
    )
    result = get_frame_annotations(fake_client(task), 1, 3)
    assert (result.frame, result.width, result.height) == (3, 640, 480)
    assert [o.id for o in result.objects] == [1, 2]
    first, second = result.objects
    assert first.label == "car" and first.bbox == [10, 20, 110, 220]
    assert first.attributes == {"parked": "true"}
    assert second.label == "person" and second.source == "auto" and second.occluded


def test_empty_frame():
    result = get_frame_annotations(fake_client(fake_task([shape(1, 1, 4, [1, 1, 2, 2])])), 1, 3)
    assert result.objects == []


def test_tracks_on_the_frame_are_counted_as_skipped():
    track = Obj(shapes=[Obj(frame=0, outside=False), Obj(frame=10, outside=True)])
    task = fake_task([], tracks=[track])
    assert get_frame_annotations(fake_client(task), 1, 3).skipped_tracks == 1
    assert get_frame_annotations(fake_client(task), 1, 12).skipped_tracks == 0


def test_frame_outside_task():
    with pytest.raises(ValueError, match=r"0\.\.39"):
        get_frame_annotations(fake_client(fake_task([])), 1, 40)


def test_video_task_has_one_meta_entry():
    task = fake_task([], meta_frames=[Obj(width=1920, height=1080)])
    result = get_frame_annotations(fake_client(task), 1, 25)
    assert (result.width, result.height) == (1920, 1080)
