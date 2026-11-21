from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

from cvat_sdk import Client
from cvat_sdk.api_client import models
from pydantic import BaseModel, ConfigDict, Field

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.annotations import check_frame
from cvat_mcp.services.issues import check_bbox, find_job_for_frame

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


@contextmanager
def _item(index: int) -> Iterator[None]:
    """Name the list item an argument error is about."""
    try:
        yield
    except InvalidArgument as exc:
        raise InvalidArgument(f"objects[{index}]: {exc}") from exc


def _check_unique(refs: list[tuple[str, int]]) -> None:
    seen = set()
    for kind, id_ in refs:
        if (kind, id_) in seen:
            raise InvalidArgument(f"{kind} #{id_} is listed twice")
        seen.add((kind, id_))


def _not_found(kind: str, ids) -> InvalidArgument:
    return InvalidArgument(
        f"{kind} id(s) {', '.join(map(str, ids))} not among this task's {kind}s (tracks are "
        "read-only and not listed here); use get_frame_annotations for the ids on a frame"
    )


def _by_id(task) -> dict[str, dict]:
    """The task's shapes and tags by id; the two id sequences are separate in CVAT."""
    annotations = task.get_annotations()
    return {
        "shape": {s.id: s for s in annotations.shapes},
        "tag": {t.id: t for t in annotations.tags},
    }


def _attribute_requests(values: list[dict]) -> list[models.AttributeValRequest]:
    return [models.AttributeValRequest(**v) for v in values]


def _patch(client: Client, action: str, task_id: int, shapes: list, tags: list):
    # The SDK's update_annotations() drops the response, which holds the new ids.
    data, _ = client.api_client.tasks_api.partial_update_annotations(
        action,
        task_id,
        patched_labeled_data_request=models.PatchedLabeledDataRequest(
            shapes=shapes, tags=tags, tracks=[]
        ),
    )
    return data


def _key(obj, kind: str) -> tuple:
    if kind == "tag":
        return (kind, obj.frame, obj.label_id)
    return (kind, obj.frame, obj.label_id, str(obj.type), tuple(round(p, 2) for p in obj.points))


def create_annotations(
    client: Client, base_url: str, task_id: int, objects: list[NewObject]
) -> AnnotationEdit:
    task = client.tasks.retrieve(task_id)
    check_editable(task)
    labels = task.get_labels()
    sent = []  # (kind, request) in item order
    for index, obj in enumerate(objects):
        with _item(index):
            check_frame(obj.frame, task.size)
            label = label_for(labels, obj.label, obj.type)
            points = check_geometry(obj.type, obj.bbox, obj.points)
            attributes = _attribute_requests(attribute_values(label, obj.attributes, {}))
        common = {"frame": obj.frame, "label_id": label.id, "attributes": attributes}
        if obj.type == "tag":
            sent.append(("tag", models.LabeledImageRequest(**common)))
        else:
            request = models.LabeledShapeRequest(
                type=obj.type, points=points, occluded=obj.occluded, **common
            )
            sent.append(("shape", request))

    data = _patch(client, "create", task_id, *_split(sent))
    # Pair ids by content: across jobs CVAT returns the objects grouped by job, not in order.
    new_ids = defaultdict(list)
    for kind, created in (("shape", data.shapes), ("tag", data.tags)):
        for c in created:
            new_ids[_key(c, kind)].append(c.id)
    names = {label.id: label.name for label in labels}
    first = objects[0].frame
    job = find_job_for_frame(task.get_jobs(), first)
    base = f"{base_url.rstrip('/')}/tasks/{task_id}"
    return AnnotationEdit(
        task_id=task_id,
        action="created",
        objects=[
            EditedObject(
                id=new_ids[_key(r, kind)].pop(0), kind=kind, frame=r.frame, label=names[r.label_id]
            )
            for kind, r in sent
        ],
        url=f"{base}/jobs/{job.id}?frame={first}" if job else base,
    )


def _split(sent: list[tuple[str, object]]) -> tuple[list, list]:
    return [r for k, r in sent if k == "shape"], [r for k, r in sent if k == "tag"]


def update_annotations(client: Client, task_id: int, objects: list[ObjectChange]) -> AnnotationEdit:
    task = client.tasks.retrieve(task_id)
    check_editable(task)
    labels = task.get_labels()
    by_id = {label.id: label for label in labels}
    found = _by_id(task)
    _check_unique([(c.kind, c.id) for c in objects])
    sent = []
    for index, change in enumerate(objects):
        with _item(index):
            obj = found[change.kind].get(change.id)
            if obj is None:
                raise _not_found(change.kind, [change.id])
            type_ = "tag" if change.kind == "tag" else str(obj.type)
            label = label_for(labels, change.label, type_) if change.label else by_id[obj.label_id]
            # CVAT's update replaces the whole object, so send every field, changed or not.
            data = obj.to_dict()
            if change.bbox is not None or change.points is not None:
                if type_ not in ("rectangle", "tag", *MIN_POINTS):
                    raise InvalidArgument(
                        f"the geometry of a {type_} can't be changed here (label, attributes "
                        "and occluded can); use the CVAT UI"
                    )
                data["points"] = check_geometry(type_, change.bbox, change.points)
            if change.occluded is not None:
                if change.kind == "tag":
                    raise InvalidArgument("occluded is for shapes only")
                data["occluded"] = change.occluded
            current = {a.spec_id: a.value for a in obj.attributes}
            values = attribute_values(label, change.attributes or {}, current)
            data |= {"label_id": label.id, "attributes": _attribute_requests(values)}
        model = models.LabeledImageRequest if change.kind == "tag" else models.LabeledShapeRequest
        sent.append((change.kind, model(**data)))

    _patch(client, "update", task_id, *_split(sent))
    names = {label.id: label.name for label in labels}
    return AnnotationEdit(
        task_id=task_id,
        action="updated",
        objects=[
            EditedObject(id=r.id, kind=kind, frame=r.frame, label=names[r.label_id])
            for kind, r in sent
        ],
    )


def delete_annotations(
    client: Client, task_id: int, shape_ids: list[int], tag_ids: list[int]
) -> AnnotationEdit:
    """Delete objects named by explicit ids, all or none.

    Never uses the SDK's remove_annotations(): an empty id list there deletes every annotation.
    """
    if not shape_ids and not tag_ids:
        raise InvalidArgument("pass at least one id in shape_ids or tag_ids")
    refs = [("shape", i) for i in shape_ids] + [("tag", i) for i in tag_ids]
    _check_unique(refs)
    task = client.tasks.retrieve(task_id)
    check_editable(task)
    found = _by_id(task)
    for kind, ids in (("shape", shape_ids), ("tag", tag_ids)):
        missing = [i for i in ids if i not in found[kind]]
        if missing:
            raise _not_found(kind, missing)
    model = {"shape": models.LabeledShapeRequest, "tag": models.LabeledImageRequest}
    sent = [(kind, model[kind](**found[kind][i].to_dict())) for kind, i in refs]

    _patch(client, "delete", task_id, *_split(sent))
    names = {label.id: label.name for label in task.get_labels()}
    return AnnotationEdit(
        task_id=task_id,
        action="deleted",
        objects=[
            EditedObject(id=r.id, kind=kind, frame=r.frame, label=names[r.label_id])
            for kind, r in sent
        ],
    )
