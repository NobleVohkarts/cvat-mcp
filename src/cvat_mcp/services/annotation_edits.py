import json
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Literal

from cvat_sdk import Client
from cvat_sdk.api_client import models
from cvat_sdk.attributes import attribute_vals_from_dict, attribute_value_validator
from pydantic import BaseModel, ConfigDict, Field

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.annotations import check_frame, check_job_frame, job_scope
from cvat_mcp.services.issues import check_bbox, find_job_for_frame, job_url

ShapeType = Literal["rectangle", "polygon", "polyline", "points"]
MIN_POINTS = {"polygon": 3, "polyline": 2, "points": 1}
REQUEST = {"shape": models.LabeledShapeRequest, "tag": models.LabeledImageRequest}

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
    id: int | None  # None: CVAT's answer could not be matched; see get_frame_annotations
    kind: str
    frame: int
    label: str


class AnnotationEdit(BaseModel):
    task_id: int
    action: str
    objects: list[EditedObject]
    url: str | None = None
    job_id: int | None = None  # set when one job's annotations were edited (e.g. ground truth)


def check_editable(task, job_id: int | None = None) -> None:
    if str(task.validation_mode) == "gt_pool":
        # CVAT's task annotations include the ground-truth job in this mode
        raise InvalidArgument(
            f"task {task.id} uses a ground-truth pool; its annotations are read-only here, "
            "use the CVAT UI"
        )
    # shortcut: a task PATCH writes overlap frames into both jobs and deletes only one copy;
    # upgrade = job-level PATCH per frame's job. One job's PATCH (job_id) has no such problem.
    if task.overlap and job_id is None:
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


def check_value(spec, value: str) -> None:
    if attribute_value_validator(spec)(value):
        return
    kind = spec.input_type.value
    if kind == "checkbox":
        allowed = "true or false"
    elif kind == "number":
        allowed = "a whole number in {}..{}, step {}".format(*spec.values)
    else:
        allowed = f"one of {', '.join(spec.values)}"
    raise InvalidArgument(f"attribute {spec.name!r} must be {allowed}")


def attribute_values(
    label, given: dict[str, str], current: dict[int, str]
) -> list[models.AttributeValRequest]:
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
        check_value(specs[name], value)
        values[specs[name].id] = value
    return attribute_vals_from_dict(
        {a.id: values[a.id] for a in label.attributes if a.id in values}
    )


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


def _not_found(kind: str, ids, job_id: int | None = None) -> InvalidArgument:
    scope = f"job {job_id}'s" if job_id is not None else "this task's"
    return InvalidArgument(
        f"{kind} id(s) {', '.join(map(str, ids))} not among {scope} {kind}s (tracks are "
        "read-only and not listed here); use get_frame_annotations for the ids on a frame"
    )


def _by_id(task) -> dict[str, dict]:
    """The task's shapes and tags by id; the two id sequences are separate in CVAT."""
    annotations = task.get_annotations()
    return {
        "shape": {s.id: s for s in annotations.shapes},
        "tag": {t.id: t for t in annotations.tags},
    }


Sent = list[tuple[str, object]]  # (kind, request) in item order


def _patch(client: Client, action: str, task_id: int, sent: Sent, job_id: int | None = None):
    # The SDK's update_annotations() drops the response, which holds the new ids.
    api = client.api_client.jobs_api if job_id is not None else client.api_client.tasks_api
    data, response = api.partial_update_annotations(
        action,
        job_id if job_id is not None else task_id,
        patched_labeled_data_request=models.PatchedLabeledDataRequest(
            shapes=[r for k, r in sent if k == "shape"],
            tags=[r for k, r in sent if k == "tag"],
            tracks=[],
        ),
    )
    if job_id is not None:  # the SDK's schema has no body for the job PATCH, but CVAT sends one
        data = json.loads(response.data, object_hook=lambda d: SimpleNamespace(**d))
    return data


def _result(
    action: str,
    task_id: int,
    labels,
    sent: Sent,
    ids=None,
    url: str | None = None,
    job_id: int | None = None,
) -> AnnotationEdit:
    names = {label.id: label.name for label in labels}
    return AnnotationEdit(
        task_id=task_id,
        action=action,
        objects=[
            EditedObject(
                id=ids[i] if ids is not None else r.id,
                kind=kind,
                frame=r.frame,
                label=names[r.label_id],
            )
            for i, (kind, r) in enumerate(sent)
        ],
        url=url,
        job_id=job_id,
    )


def _source(client: Client, task, job_id: int | None):
    """Where the annotations live: one job of the task (job_id), or the task's annotation jobs."""
    return job_scope(client, task, job_id)[0] if job_id is not None else task


def _key(obj, kind: str) -> tuple:
    # Not attributes: CVAT may answer with defaults filled in, and then nothing would match.
    if kind == "tag":
        return (kind, obj.frame, obj.label_id)
    points = tuple(round(p, 2) for p in obj.points)
    return (kind, obj.frame, obj.label_id, str(obj.type), points, obj.occluded)


def create_annotations(
    client: Client,
    base_url: str,
    task_id: int,
    objects: list[NewObject],
    job_id: int | None = None,
) -> AnnotationEdit:
    # Argument checks first: they need no requests.
    points = []
    for index, obj in enumerate(objects):
        with _item(index):
            points.append(check_geometry(obj.type, obj.bbox, obj.points))
    task = client.tasks.retrieve(task_id)
    check_editable(task, job_id)
    job, job_frames = job_scope(client, task, job_id) if job_id is not None else (None, None)
    labels = task.get_labels()
    sent = []
    for index, (obj, obj_points) in enumerate(zip(objects, points)):
        with _item(index):
            check_frame(obj.frame, task.size)
            if job is not None:
                check_job_frame(obj.frame, job, job_frames)
            label = label_for(labels, obj.label, obj.type)
            attributes = attribute_values(label, obj.attributes, {})
        common = {"frame": obj.frame, "label_id": label.id, "attributes": attributes}
        if obj.type == "tag":
            sent.append(("tag", models.LabeledImageRequest(**common)))
        else:
            request = models.LabeledShapeRequest(
                type=obj.type, points=obj_points, occluded=obj.occluded, **common
            )
            sent.append(("shape", request))

    first = objects[0].frame
    if job is None:  # before the write: nothing may fail after it
        job = find_job_for_frame(task.get_jobs(), first)

    data = _patch(client, "create", task_id, sent, job_id)
    # Pair ids by content: across jobs CVAT returns the objects grouped by job, not in order.
    new_ids = defaultdict(list)
    for kind, created in (("shape", data.shapes), ("tag", data.tags)):
        for c in created:
            new_ids[_key(c, kind)].append(c.id)
    return _result(
        "created",
        task_id,
        labels,
        sent,
        ids=[(new_ids[_key(r, kind)] or [None]).pop(0) for kind, r in sent],
        url=job_url(base_url, task_id, job.id, first) if job else job_url(base_url, task_id),
        job_id=job_id,
    )


def update_annotations(
    client: Client, task_id: int, objects: list[ObjectChange], job_id: int | None = None
) -> AnnotationEdit:
    # Argument checks first: they need no requests.
    _check_unique([(c.kind, c.id) for c in objects])
    for index, change in enumerate(objects):
        with _item(index):
            if change.kind == "tag":
                check_geometry("tag", change.bbox, change.points)
                if change.occluded is not None:
                    raise InvalidArgument("occluded is for shapes only")
    task = client.tasks.retrieve(task_id)
    check_editable(task, job_id)
    labels = task.get_labels()
    label_by_id = {label.id: label for label in labels}
    found = _by_id(_source(client, task, job_id))
    sent = []
    for index, change in enumerate(objects):
        with _item(index):
            obj = found[change.kind].get(change.id)
            if obj is None:
                raise _not_found(change.kind, [change.id], job_id)
            type_ = "tag" if change.kind == "tag" else str(obj.type)
            if change.label is not None and type_ == "skeleton":
                # its elements carry the old label's sublabels
                raise InvalidArgument("a skeleton's label can't be changed here; use the CVAT UI")
            label = (
                label_by_id[obj.label_id]
                if change.label is None
                else label_for(labels, change.label, type_)
            )
            # CVAT's update replaces the whole object, so send every field, changed or not.
            data = obj.to_dict()
            if change.kind == "shape" and (change.bbox is not None or change.points is not None):
                if type_ not in ("rectangle", *MIN_POINTS):
                    raise InvalidArgument(
                        f"the geometry of a {type_} can't be changed here (label, attributes "
                        "and occluded can); use the CVAT UI"
                    )
                data["points"] = check_geometry(type_, change.bbox, change.points)
            if change.occluded is not None:
                data["occluded"] = change.occluded
            current = {a.spec_id: a.value for a in obj.attributes}
            data |= {
                "label_id": label.id,
                "attributes": attribute_values(label, change.attributes or {}, current),
            }
        sent.append((change.kind, REQUEST[change.kind](**data)))

    _patch(client, "update", task_id, sent, job_id)
    return _result("updated", task_id, labels, sent, job_id=job_id)


def delete_annotations(
    client: Client,
    task_id: int,
    shape_ids: list[int],
    tag_ids: list[int],
    job_id: int | None = None,
) -> AnnotationEdit:
    """Delete objects named by explicit ids, all or none.

    Never uses the SDK's remove_annotations(): an empty id list there deletes every annotation.
    """
    if not shape_ids and not tag_ids:
        raise InvalidArgument("pass at least one id in shape_ids or tag_ids")
    refs = [("shape", i) for i in shape_ids] + [("tag", i) for i in tag_ids]
    _check_unique(refs)
    task = client.tasks.retrieve(task_id)
    check_editable(task, job_id)
    found = _by_id(_source(client, task, job_id))
    for kind, ids in (("shape", shape_ids), ("tag", tag_ids)):
        missing = [i for i in ids if i not in found[kind]]
        if missing:
            raise _not_found(kind, missing, job_id)
    sent = [(kind, REQUEST[kind](**found[kind][i].to_dict())) for kind, i in refs]

    _patch(client, "delete", task_id, sent, job_id)
    return _result("deleted", task_id, task.get_labels(), sent, job_id=job_id)
