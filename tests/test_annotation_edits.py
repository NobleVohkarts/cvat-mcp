import pytest
from fakes import LABELS, Obj, fake_task

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.annotation_edits import (
    attribute_values,
    check_editable,
    check_geometry,
    label_for,
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


@pytest.mark.parametrize(
    "label, type_", [(TAG_LABEL, "rectangle"), (CAR, "polygon"), (CAR, "tag")]
)
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
