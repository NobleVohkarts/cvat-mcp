from unittest.mock import Mock

import pytest
from fakes import LABELS, Obj, fake_client, fake_task, shape, tag

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.annotation_edits import (
    NewObject,
    ObjectChange,
    attribute_values,
    check_editable,
    check_geometry,
    create_annotations,
    delete_annotations,
    label_for,
    update_annotations,
)

CAR = LABELS[0]
TAG_LABEL = Obj(id=3, name="night", type="tag", attributes=[])
ANY_LABEL = Obj(id=4, name="thing", type="any", attributes=[])
SPEED = Obj(id=12, name="speed", input_type="number", values=["0", "10", "1"])


@pytest.mark.parametrize(
    "type_, bbox, points, expected",
    [
        ("rectangle", [1, 2, 3, 4], None, [1, 2, 3, 4]),
        ("polygon", None, [0, 0, 5, 0, 5, 5], [0, 0, 5, 0, 5, 5]),
        ("polyline", None, [0, 0, 5, 5], [0, 0, 5, 5]),
        ("points", None, [1, 1], [1, 1]),
        ("tag", None, None, []),
    ],
)
def test_check_geometry_accepts(type_, bbox, points, expected):
    assert check_geometry(type_, bbox, points) == expected


@pytest.mark.parametrize(
    "type_, bbox, points, match",
    [
        ("rectangle", None, [1, 2, 3, 4], "bbox"),
        ("rectangle", None, None, "bbox"),
        ("polygon", [1, 2, 3, 4], None, "points"),
        ("polygon", [1, 2, 3, 4], [0, 0, 5, 0, 5, 5], "one of"),
        ("polyline", None, [0, 0, 5], "pairs"),
        ("polygon", None, [0, 0, 5, 5], "at least 3"),
        ("polyline", None, [0, 0], "at least 2"),
        ("rectangle", [3, 2, 1, 4], None, "x1 < x2"),
        ("tag", [1, 2, 3, 4], None, "no geometry"),
    ],
)
def test_check_geometry_refuses(type_, bbox, points, match):
    with pytest.raises(InvalidArgument, match=match):
        check_geometry(type_, bbox, points)


def test_label_for_unknown_lists_labels():
    with pytest.raises(InvalidArgument, match="car, person"):
        label_for(LABELS, "bus", "rectangle")


@pytest.mark.parametrize("label, type_", [(TAG_LABEL, "rectangle"), (CAR, "polygon"), (CAR, "tag")])
def test_label_for_refuses_other_type(label, type_):
    with pytest.raises(InvalidArgument, match=label.name):
        label_for([label], label.name, type_)


@pytest.mark.parametrize("type_", ["rectangle", "polygon", "polyline", "points", "tag"])
def test_label_for_any_accepts_every_type(type_):
    assert label_for([ANY_LABEL], "thing", type_) is ANY_LABEL


def test_label_for_tag_label():
    assert label_for([TAG_LABEL], "night", "tag") is TAG_LABEL


def test_attribute_values_by_name():
    assert attribute_values(CAR, {"color": "red", "parked": "true"}, {}) == [
        {"spec_id": 10, "value": "true"},
        {"spec_id": 11, "value": "red"},
    ]


@pytest.mark.parametrize(
    "given, match",
    [
        ({"size": "big"}, "parked, color"),
        ({"color": "green"}, "red, blue"),
        ({"parked": "yes"}, "true or false"),
    ],
)
def test_attribute_values_refuses(given, match):
    with pytest.raises(InvalidArgument, match=match):
        attribute_values(CAR, given, {})


@pytest.mark.parametrize("value", ["11", "-1", "fast"])
def test_number_attribute_outside_range(value):
    label = Obj(id=5, name="truck", type="rectangle", attributes=[SPEED])
    with pytest.raises(InvalidArgument, match=r"0\.\.10"):
        attribute_values(label, {"speed": value}, {})


def test_attribute_values_keep_current_and_drop_foreign():
    # 11 is kept, 10 overridden by name, 99 (another label's attribute) dropped
    current = {10: "false", 11: "blue", 99: "x"}
    assert attribute_values(CAR, {"parked": "true"}, current) == [
        {"spec_id": 10, "value": "true"},
        {"spec_id": 11, "value": "blue"},
    ]


def test_check_editable_refuses_overlap():
    with pytest.raises(InvalidArgument, match="overlap"):
        check_editable(fake_task(overlap=5))


@pytest.mark.parametrize("overlap", [0, None])
def test_check_editable_without_overlap(overlap):
    check_editable(fake_task(overlap=overlap))


JOBS = [
    Obj(id=2, type="annotation", start_frame=0, stop_frame=19),
    Obj(id=3, type="annotation", start_frame=20, stop_frame=39),
]
PARKED = Obj(spec_id=10, value="true")


def _client(response=None, shapes=(), tags=(), overlap=0):
    api = Mock()
    api.partial_update_annotations.return_value = (response, None)
    task = fake_task(shapes, tags=tags, jobs=JOBS, overlap=overlap)
    task.get_labels = lambda: [*LABELS, TAG_LABEL]
    return fake_client(task, api_client=Obj(tasks_api=api)), api


def _sent(api, action):
    (call,) = api.partial_update_annotations.call_args_list
    assert call.args == (action, 1)
    return call.kwargs["patched_labeled_data_request"]


def _created(id, frame, label_id, points, type="rectangle"):  # pylint: disable=redefined-builtin
    return Obj(id=id, frame=frame, label_id=label_id, type=type, points=points)


def test_create_sends_one_patch_and_returns_ids():
    response = Obj(
        shapes=[_created(101, 3, 1, [10, 20, 110, 220])], tags=[Obj(id=7, frame=3, label_id=3)]
    )
    client, api = _client(response)
    objects = [
        NewObject(frame=3, label="car", bbox=[10, 20, 110, 220], attributes={"color": "red"}),
        NewObject(frame=3, label="night", type="tag"),
    ]
    result = create_annotations(client, "http://cvat/", 1, objects)
    sent = _sent(api, "create")
    (box,) = sent.shapes
    assert (box.label_id, box.frame, box.points, str(box.type)) == (
        1,
        3,
        [10, 20, 110, 220],
        "rectangle",
    )
    assert [(a.spec_id, a.value) for a in box.attributes] == [(11, "red")]
    assert [(t.frame, t.label_id) for t in sent.tags] == [(3, 3)]
    assert [(o.id, o.kind, o.frame, o.label) for o in result.objects] == [
        (101, "shape", 3, "car"),
        (7, "tag", 3, "night"),
    ]
    assert result.url == "http://cvat/tasks/1/jobs/2?frame=3"


def test_create_pairs_ids_by_content_not_order():
    response = Obj(
        shapes=[_created(202, 25, 2, [5, 5, 9, 9]), _created(201, 3, 1, [1, 1, 4, 4])], tags=[]
    )
    client, _ = _client(response)
    objects = [
        NewObject(frame=3, label="car", bbox=[1, 1, 4, 4]),
        NewObject(frame=25, label="person", bbox=[5, 5, 9, 9]),
    ]
    result = create_annotations(client, "http://cvat", 1, objects)
    assert [o.id for o in result.objects] == [201, 202]


@pytest.mark.parametrize(
    "bad, match",
    [
        (NewObject(frame=3, label="bus", bbox=[1, 1, 4, 4]), r"objects\[1\]: unknown label"),
        (NewObject(frame=40, label="car", bbox=[1, 1, 4, 4]), r"objects\[1\]: frame 40"),
        (NewObject(frame=3, label="car", points=[1, 1, 4, 4]), r"objects\[1\]: .*bbox"),
    ],
)
def test_one_bad_item_blocks_the_batch(bad, match):
    client, api = _client()
    with pytest.raises(InvalidArgument, match=match):
        create_annotations(
            client, "http://cvat", 1, [NewObject(frame=3, label="car", bbox=[1, 1, 4, 4]), bad]
        )
    api.partial_update_annotations.assert_not_called()


def test_update_merges_and_keeps_other_fields():
    box = shape(1, 1, 3, [10, 20, 110, 220], z_order=2, attributes=[PARKED])
    client, api = _client(shapes=[box])
    result = update_annotations(client, 1, [ObjectChange(id=1, bbox=[0, 0, 50, 50])])
    (sent,) = _sent(api, "update").shapes
    assert (sent.id, sent.z_order, sent.points) == (1, 2, [0, 0, 50, 50])
    assert [(a.spec_id, a.value) for a in sent.attributes] == [(10, "true")]
    assert [(o.id, o.kind, o.frame, o.label) for o in result.objects] == [(1, "shape", 3, "car")]


def test_update_label_drops_old_attributes():
    client, api = _client(shapes=[shape(1, 1, 3, [10, 20, 110, 220], attributes=[PARKED])])
    result = update_annotations(client, 1, [ObjectChange(id=1, label="person")])
    (sent,) = _sent(api, "update").shapes
    assert (sent.label_id, list(sent.attributes)) == (2, [])
    assert result.objects[0].label == "person"


def test_update_tag_attributes():
    client, api = _client(tags=[tag(7, 1, 3)])
    update_annotations(client, 1, [ObjectChange(id=7, kind="tag", attributes={"parked": "true"})])
    (sent,) = _sent(api, "update").tags
    assert sent.id == 7 and [(a.spec_id, a.value) for a in sent.attributes] == [(10, "true")]


@pytest.mark.parametrize(
    "changes, match",
    [
        ([ObjectChange(id=9)], "not among this task's shapes"),
        ([ObjectChange(id=7)], "not among this task's shapes"),  # 7 is only a tag
        ([ObjectChange(id=1, label="car"), ObjectChange(id=1, occluded=True)], "twice"),
        ([ObjectChange(id=1, bbox=[0, 0, 5, 5])], "points"),  # #1 is a polygon here
        ([ObjectChange(id=2, points=[0, 0, 5, 5, 9, 9])], "geometry"),  # mask
        ([ObjectChange(id=7, kind="tag", occluded=True)], "shapes only"),
        ([ObjectChange(id=1, label="night")], "is for tag objects"),
    ],
)
def test_update_refusals(changes, match):
    shapes = [
        shape(1, 1, 3, [0, 0, 5, 0, 5, 5], type="polygon"),
        shape(2, 1, 3, [1, 2, 0, 0, 3, 3], type="mask"),
    ]
    client, api = _client(shapes=shapes, tags=[tag(7, 3, 3)])
    with pytest.raises(InvalidArgument, match=match):
        update_annotations(client, 1, changes)
    api.partial_update_annotations.assert_not_called()


def _delete_client():
    client, api = _client(shapes=[shape(1, 1, 3, [1, 1, 4, 4])], tags=[tag(7, 3, 3)])
    client.tasks.retrieve(1).remove_annotations = Mock()
    return client, api


def test_delete_sends_fetched_objects():
    client, api = _delete_client()
    result = delete_annotations(client, 1, [1], [7])
    sent = _sent(api, "delete")
    assert [s.id for s in sent.shapes] == [1] and [t.id for t in sent.tags] == [7]
    assert [(o.id, o.kind, o.frame, o.label) for o in result.objects] == [
        (1, "shape", 3, "car"),
        (7, "tag", 3, "night"),
    ]
    client.tasks.retrieve(1).remove_annotations.assert_not_called()


@pytest.mark.parametrize(
    "shape_ids, tag_ids, match",
    [
        ([], [], "at least one"),
        ([1, 2, 3], [], "2, 3"),
        ([1, 1], [], "twice"),
        ([7], [], "not among this task's shapes"),  # 7 is only a tag id
        ([], [1], "not among this task's tags"),  # 1 is only a shape id
    ],
)
def test_delete_refusals(shape_ids, tag_ids, match):
    client, api = _delete_client()
    with pytest.raises(InvalidArgument, match=match):
        delete_annotations(client, 1, shape_ids, tag_ids)
    api.partial_update_annotations.assert_not_called()


@pytest.mark.parametrize(
    "edit",
    [
        lambda c: create_annotations(
            c, "http://cvat", 1, [NewObject(frame=3, label="car", bbox=[1, 1, 4, 4])]
        ),
        lambda c: update_annotations(c, 1, [ObjectChange(id=1, occluded=True)]),
        lambda c: delete_annotations(c, 1, [1], []),
    ],
)
def test_overlap_task_refused(edit):
    client, api = _client(shapes=[shape(1, 1, 3, [1, 1, 4, 4])], overlap=5)
    with pytest.raises(InvalidArgument, match="overlap"):
        edit(client)
    api.partial_update_annotations.assert_not_called()
