from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.issues import check_bbox

ShapeType = Literal["rectangle", "polygon", "polyline", "points"]
MIN_POINTS = {"polygon": 3, "polyline": 2, "points": 1}

BBOX = Field(None, description="[x1, y1, x2, y2] in original pixels; rectangles only")
POINTS = Field(
    None, description="flat [x1, y1, x2, y2, ...] in original pixels; polygon, polyline, points"
)


class NewObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame: int = Field(description="task frame number (0-based)")
    label: str = Field(description="label name, one of the task's labels")
    type: ShapeType | Literal["tag"] = "rectangle"
    bbox: list[float] | None = BBOX
    points: list[float] | None = POINTS
    attributes: dict[str, str] = Field({}, description="attribute name -> value")
    occluded: bool = False


class ObjectChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(description="shape or tag id from get_frame_annotations")
    kind: Literal["shape", "tag"] = Field("shape", description="shape and tag ids are separate")
    label: str | None = Field(None, description="new label name")
    bbox: list[float] | None = BBOX
    points: list[float] | None = POINTS
    attributes: dict[str, str] | None = Field(
        None, description="attribute name -> value; merged into the current values"
    )
    occluded: bool | None = Field(None, description="shapes only")


class EditedObject(BaseModel):
    id: int
    kind: str
    frame: int
    label: str


class AnnotationEdit(BaseModel):
    task_id: int
    action: str
    objects: list[EditedObject]
    url: str | None = None


def check_editable(task) -> None:
    # shortcut: a task PATCH writes overlap frames into both jobs and deletes only one copy;
    # upgrade = job-level PATCH per frame's job.
    if task.overlap:
        raise InvalidArgument(
            f"task {task.id} has overlapping jobs (overlap {task.overlap}); editing its "
            "annotations is not supported here, use the CVAT UI"
        )


def check_geometry(type: str, bbox, points) -> list[float]:  # pylint: disable=redefined-builtin
    """The points to send for an object of this type, from exactly one of bbox or points."""
    if type == "tag":
        if bbox is not None or points is not None:
            raise InvalidArgument("a tag has no geometry: pass no bbox or points")
        return []
    if bbox is not None and points is not None:
        raise InvalidArgument("pass one of bbox or points, not both")
    if type == "rectangle":
        if bbox is None:
            raise InvalidArgument("a rectangle needs bbox [x1, y1, x2, y2], not points")
        check_bbox(bbox)
        return list(bbox)
    if points is None:
        raise InvalidArgument(f"a {type} needs points [x1, y1, x2, y2, ...], not bbox")
    if len(points) % 2:
        raise InvalidArgument("points must be x, y pairs (an even count)")
    if len(points) // 2 < MIN_POINTS[type]:
        raise InvalidArgument(f"a {type} needs at least {MIN_POINTS[type]} points")
    return list(points)


def label_for(labels, name: str, type: str):  # pylint: disable=redefined-builtin
    """The task label with this name, if it can be used for an object of this type."""
    label = next((label for label in labels if label.name == name), None)
    if label is None:
        raise InvalidArgument(
            f"unknown label {name!r}; the task's labels: {', '.join(l.name for l in labels)}"
        )
    label_type = str(label.type)
    if label_type not in ("any", type):
        raise InvalidArgument(f"label {name!r} is for {label_type} objects, not {type}")
    return label


def _check_value(spec, value: str) -> None:
    kind = str(spec.input_type)
    if kind == "checkbox" and value not in ("true", "false"):
        raise InvalidArgument(f"attribute {spec.name!r} is a checkbox: true or false")
    if kind in ("select", "radio") and value not in spec.values:
        raise InvalidArgument(f"attribute {spec.name!r} must be one of {', '.join(spec.values)}")
    if kind == "number":
        low, high = float(spec.values[0]), float(spec.values[1])
        try:
            ok = low <= float(value) <= high
        except ValueError:
            ok = False
        if not ok:
            raise InvalidArgument(
                f"attribute {spec.name!r} is a number in {spec.values[0]}..{spec.values[1]}"
            )


def attribute_values(label, given: dict[str, str], current: dict[int, str]) -> list[dict]:
    """The label's attribute values: current ones (by spec id) overridden by `given` (by name).

    Values of attributes that are not the label's are dropped (CVAT refuses them)."""
    specs = {a.name: a for a in label.attributes}
    unknown = sorted(set(given) - set(specs))
    if unknown:
        raise InvalidArgument(
            f"unknown attribute(s) {', '.join(unknown)} for label {label.name!r}; valid: "
            f"{', '.join(specs) or 'none'}"
        )
    values = {a.id: current[a.id] for a in label.attributes if a.id in current}
    for name, value in given.items():
        _check_value(specs[name], value)
        values[specs[name].id] = value
    return [{"spec_id": a.id, "value": values[a.id]} for a in label.attributes if a.id in values]
