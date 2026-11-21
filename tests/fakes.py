"""Tiny stand-ins for cvat-sdk objects: only the attributes our services read."""

from types import SimpleNamespace as Obj


def _with_to_dict(obj):
    obj.to_dict = lambda: {k: v for k, v in vars(obj).items() if k != "to_dict"}
    return obj


def shape(id, label_id, frame, points, **kw):  # pylint: disable=redefined-builtin
    defaults = {"type": "rectangle", "attributes": [], "source": "manual", "occluded": False}
    return _with_to_dict(
        Obj(id=id, label_id=label_id, frame=frame, points=points, **{**defaults, **kw})
    )


def tag(id, label_id, frame, **kw):  # pylint: disable=redefined-builtin
    defaults = {"attributes": [], "source": "manual", "group": 0}
    return _with_to_dict(Obj(id=id, label_id=label_id, frame=frame, **{**defaults, **kw}))


LABELS = [
    Obj(
        id=1, name="car", color="#ff0000", type="rectangle", attributes=[Obj(id=10, name="parked")]
    ),
    Obj(id=2, name="person", color="#00ff00", type="rectangle", attributes=[]),
]


def fake_task(
    shapes=(), size=40, tracks=(), tags=(), jobs=(), task_id=1, meta_frames=None, overlap=0, **kw
):
    return Obj(
        id=task_id,
        size=size,
        overlap=overlap,
        get_annotations=lambda: Obj(shapes=list(shapes), tracks=list(tracks), tags=list(tags)),
        get_labels=lambda: LABELS,
        get_meta=lambda: Obj(
            frames=meta_frames or [Obj(width=640, height=480) for _ in range(size)]
        ),
        get_jobs=lambda: list(jobs),
        **kw,
    )


def fake_client(task, **repos):
    return Obj(tasks=Obj(retrieve=lambda _id: task), **repos)
