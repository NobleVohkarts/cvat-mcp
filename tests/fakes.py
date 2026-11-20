"""Tiny stand-ins for cvat-sdk objects: only the attributes our services read."""


class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def shape(id, label_id, frame, points, **kw):  # pylint: disable=redefined-builtin
    defaults = {"type": "rectangle", "attributes": [], "source": "manual", "occluded": False}
    return Obj(id=id, label_id=label_id, frame=frame, points=points, **{**defaults, **kw})


def fake_task(shapes, size=40, tracks=(), jobs=(), task_id=1):
    labels = [
        Obj(id=1, name="car", color="#ff0000", attributes=[Obj(id=10, name="parked")]),
        Obj(id=2, name="person", color="#00ff00", attributes=[]),
    ]
    return Obj(
        id=task_id,
        size=size,
        get_annotations=lambda: Obj(shapes=list(shapes), tracks=list(tracks), tags=[]),
        get_labels=lambda: labels,
        get_meta=lambda: Obj(frames=[Obj(width=640, height=480) for _ in range(size)]),
        get_jobs=lambda: list(jobs),
    )


def fake_client(task, **repos):
    return Obj(tasks=Obj(retrieve=lambda _id: task), **repos)
