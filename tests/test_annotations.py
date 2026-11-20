import pytest
from fakes import Obj, fake_client, fake_task, shape

from cvat_mcp.services.annotations import check_frame, get_frame_annotations, shape_bbox


def test_shape_bbox_rectangle():
    assert shape_bbox([10, 20, 110, 220]) == [10, 20, 110, 220]


def test_shape_bbox_polygon():
    assert shape_bbox([5, 50, 30, 10, 60, 40]) == [5, 10, 60, 50]


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
