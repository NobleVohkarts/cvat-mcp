import pytest
from fakes import Obj, fake_client, fake_task, gt_job, job_client, shape, tag

from cvat_mcp.services.annotations import check_frame, get_frame_annotations, shape_bbox
from cvat_mcp.services.tasks import task_frames


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


def test_polygon_has_points_rectangle_has_none():
    polygon = shape(1, 1, 3, [5, 50, 30, 10, 60, 40], type="polygon")
    task = fake_task([polygon, shape(2, 1, 3, [10, 20, 110, 220])])
    first, second = get_frame_annotations(fake_client(task), 1, 3).objects
    assert first.points == [5, 50, 30, 10, 60, 40]
    assert second.points is None


def test_tags_of_that_frame():
    tags = [tag(7, 1, 3, attributes=[Obj(spec_id=10, value="true")]), tag(8, 2, 4)]
    result = get_frame_annotations(fake_client(fake_task([], tags=tags)), 1, 3)
    assert [t.id for t in result.tags] == [7]
    assert result.tags[0].label == "car"
    assert result.tags[0].attributes == {"parked": "true"}


@pytest.mark.parametrize("type_", ["mask", "ellipse", "cuboid", "skeleton"])
def test_points_only_for_editable_types(type_):
    elements = [Obj(points=[1, 2]), Obj(points=[3, 4])]
    task = fake_task([shape(1, 1, 3, [1, 2, 3, 4, 5, 6, 7, 8], type=type_, elements=elements)])
    assert get_frame_annotations(fake_client(task), 1, 3).objects[0].points is None


def test_polyline_and_points_have_points():
    shapes = [shape(1, 1, 3, [0, 0, 5, 5], type="polyline"), shape(2, 1, 3, [1, 1], type="points")]
    result = get_frame_annotations(fake_client(fake_task(shapes)), 1, 3)
    assert [o.points for o in result.objects] == [[0, 0, 5, 5], [1, 1]]


def test_job_annotations_of_a_gt_job():
    task = fake_task([shape(1, 1, 2, [0, 0, 5, 5])])
    job = gt_job([shape(2, 2, 2, [10, 10, 50, 50])])
    result = get_frame_annotations(job_client(task, job), 1, 2, job_id=30)
    assert [o.id for o in result.objects] == [2] and result.job_id == 30


def test_job_frame_outside_gt_job():
    task = fake_task()
    with pytest.raises(ValueError, match="frame 1 is not in job 30.*0, 2"):
        get_frame_annotations(job_client(task, gt_job()), 1, 1, job_id=30)


def test_job_of_another_task():
    task = fake_task()
    with pytest.raises(ValueError, match="belongs to task 9"):
        get_frame_annotations(job_client(task, gt_job(task_id=9)), 1, 2, job_id=30)


def test_annotation_job_frames_are_its_range():
    task = fake_task()
    job = Obj(id=20, task_id=1, type="annotation", start_frame=20, stop_frame=39)
    with pytest.raises(ValueError, match="frame 5 is not in job 20"):
        get_frame_annotations(job_client(task, job), 1, 5, job_id=20)


def test_task_frames_convert_data_frames():
    meta = Obj(start_frame=10, frame_filter="step=2")
    assert task_frames(meta, [14, 10]) == [0, 2]


def test_gt_job_frames_are_task_frames():
    # data frames 10, 14 of a task starting at data frame 10 with step 2 are task frames 0, 2
    task = fake_task(start_frame=10, frame_filter="step=2")
    client = job_client(task, gt_job(included=(10, 14)))
    with pytest.raises(ValueError, match="its frames: 0, 2"):
        get_frame_annotations(client, 1, 1, job_id=30)


def test_without_job_id_nothing_changes():
    task = fake_task([shape(1, 1, 2, [0, 0, 5, 5])])
    result = get_frame_annotations(fake_client(task), 1, 2)
    assert [o.id for o in result.objects] == [1] and result.job_id is None
