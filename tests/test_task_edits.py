import pytest

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.task_edits import AttributeSpec, LabelSpec, label_requests

COLOR = AttributeSpec(name="color", input_type="select", values=["red", "blue"])


def test_label_requests_build_cvat_labels():
    (car,) = label_requests([LabelSpec(name="car", type="rectangle", attributes=[COLOR])])
    assert (car.name, str(car.type)) == ("car", "rectangle")
    (color,) = car.attributes
    assert (color.name, color.input_type.value, color.values) == (
        "color",
        "select",
        ["red", "blue"],
    )


def test_checkbox_and_text_get_default_values():
    specs = [
        AttributeSpec(name="parked", input_type="checkbox"),
        AttributeSpec(name="plate", input_type="text"),
    ]
    (label,) = label_requests([LabelSpec(name="car", attributes=specs)])
    assert [a.values for a in label.attributes] == [["false"], [""]]


@pytest.mark.parametrize(
    "specs, existing, match",
    [
        ([], (), "at least one"),
        ([LabelSpec(name="car"), LabelSpec(name="car")], (), "twice"),
        ([LabelSpec(name="car")], ("car",), "already has"),
        ([LabelSpec(name=" ")], (), "name"),
        (
            [LabelSpec(name="car", attributes=[AttributeSpec(name="c", input_type="select")])],
            (),
            "'c'",
        ),
        (
            [
                LabelSpec(
                    name="car",
                    attributes=[AttributeSpec(name="n", input_type="number", values=["0", "10"])],
                )
            ],
            (),
            "'n'",
        ),
        (
            [
                LabelSpec(
                    name="car",
                    attributes=[
                        AttributeSpec(name="n", input_type="number", values=["0", "10", "3"])
                    ],
                )
            ],
            (),
            "'n'",
        ),
        (
            [
                LabelSpec(
                    name="car",
                    attributes=[AttributeSpec(name="p", input_type="checkbox", values=["yes"])],
                )
            ],
            (),
            "'p'",
        ),
    ],
)
def test_label_requests_refuse(specs, existing, match):
    with pytest.raises(InvalidArgument, match=match):
        label_requests(specs, existing)
