"""Tiny stand-ins for cvat-sdk objects: only the attributes our services read."""

from types import SimpleNamespace as Obj
from unittest.mock import Mock


def _with_to_dict(obj):
    obj.to_dict = lambda: {k: v for k, v in vars(obj).items() if k != "to_dict"}
    return obj


def shape(id, label_id, frame, points, **kw):  # pylint: disable=redefined-builtin
    defaults = {
        "type": "rectangle",
        "attributes": [],
        "source": "manual",
        "occluded": False,
        "group": 0,
        "z_order": 0,
        "rotation": 0.0,
    }
    return _with_to_dict(
        Obj(id=id, label_id=label_id, frame=frame, points=points, **{**defaults, **kw})
    )


def tag(id, label_id, frame, **kw):  # pylint: disable=redefined-builtin
    defaults = {"attributes": [], "source": "manual", "group": 0}
    return _with_to_dict(Obj(id=id, label_id=label_id, frame=frame, **{**defaults, **kw}))


LABELS = [
    Obj(
        id=1,
        name="car",
        color="#ff0000",
        type="rectangle",
        attributes=[
            Obj(id=10, name="parked", input_type=Obj(value="checkbox"), values=["false"]),
            Obj(id=11, name="color", input_type=Obj(value="select"), values=["red", "blue"]),
        ],
    ),
    Obj(id=2, name="person", color="#00ff00", type="rectangle", attributes=[]),
]


def fake_task(
    shapes=(),
    size=40,
    tracks=(),
    tags=(),
    jobs=(),
    task_id=1,
    meta_frames=None,
    overlap=0,
    labels=tuple(LABELS),
    validation_mode=None,
    start_frame=0,
    frame_filter="",
    deleted_frames=(),
    **kw,
):
    return Obj(
        id=task_id,
        size=size,
        overlap=overlap,
        validation_mode=validation_mode,
        get_annotations=lambda: Obj(shapes=list(shapes), tracks=list(tracks), tags=list(tags)),
        get_labels=lambda: list(labels),
        get_meta=lambda: Obj(
            frames=meta_frames
            or [Obj(width=640, height=480, name=f"frame_{i:06}.jpg") for i in range(size)],
            start_frame=start_frame,
            frame_filter=frame_filter,
            deleted_frames=list(deleted_frames),
        ),
        get_jobs=lambda: list(jobs),
        **kw,
    )


def fake_client(task, **repos):
    return Obj(tasks=Obj(retrieve=lambda _id: task), **repos)


def gt_job(shapes=(), included=(0, 2), task_id=1, job_id=30, deleted=(), **kw):
    return Obj(
        id=job_id,
        task_id=task_id,
        type="ground_truth",
        start_frame=min(included),
        stop_frame=max(included),
        get_meta=lambda: Obj(included_frames=list(included), deleted_frames=list(deleted)),
        get_annotations=lambda: Obj(shapes=list(shapes), tracks=[], tags=[]),
        **kw,
    )


def annotation_job(job_id, start, stop, shapes=(), tags=(), tracks=(), task_id=1, **kw):
    """An annotation job of frames start..stop."""
    return Obj(
        id=job_id,
        task_id=task_id,
        type="annotation",
        start_frame=start,
        stop_frame=stop,
        get_annotations=lambda: Obj(shapes=list(shapes), tracks=list(tracks), tags=list(tags)),
        **kw,
    )


def job_client(task, job, **repos):
    """A fake client whose jobs.retrieve returns `job`."""
    return fake_client(task, jobs=Obj(retrieve=lambda _id: job), **repos)


def quality_requirement(
    id, name, annotation_type, enabled=False
):  # pylint: disable=redefined-builtin
    return Obj(id=id, name=name, annotation_type=annotation_type, enabled=enabled)


def fake_list_settings(
    task_id, task_reqs, project_reqs, inherit=True, project_id=9, inheriting=2, visible=True
):
    """quality_api.list_settings: by task_id the task's own settings; by project_id with
    parent_type "project" the project's (empty when not visible), with parent_type "task" the
    count of its inheriting tasks."""
    own = Obj(id=30, task_id=task_id, project_id=None, inherit=inherit, requirements=task_reqs)
    project = Obj(
        id=18, task_id=None, project_id=project_id, inherit=True, requirements=project_reqs
    )

    def list_settings(task_id=None, parent_type=None, **_):
        if task_id is not None:
            return Obj(results=[own]), None
        if parent_type == "project":
            return Obj(results=[project] if visible else []), None
        return Obj(count=inheriting, results=[]), None

    return Mock(side_effect=list_settings)


def cloud_storage(
    id,  # pylint: disable=redefined-builtin
    bucket,
    attrs="",
    owner_id=1,
    owner="admin",
    org=None,
    provider="AWS_S3_BUCKET",
    credentials="KEY_SECRET_KEY_PAIR",
    name=None,
):
    return Obj(
        id=id,
        display_name=name or f"{bucket} storage",
        provider_type=Obj(value=provider),
        resource=bucket,
        specific_attributes=attrs,
        credentials_type=Obj(value=credentials),
        owner=Obj(id=owner_id, username=owner),
        organization=org,
    )


def fake_cloudstorages_api(storages=()):
    """cloudstorages_api.list over `storages` (honouring page_size); writes are Mocks."""
    storages = list(storages)

    def list_(page_size=100, **_):
        return Obj(results=storages[:page_size], count=len(storages)), None

    return Obj(
        list=Mock(side_effect=list_),
        create=Mock(),
        partial_update=Mock(),
        destroy=Mock(),
        retrieve=lambda id, **_: (next(s for s in storages if s.id == id), None),
        retrieve_status=Mock(return_value=(None, Obj(data=b'"AVAILABLE"'))),
    )


def content_listing(*pages):
    """cloudstorages_api.retrieve_content_v2 replying with `pages` in turn, each a list of
    (name, type) pairs; every page but the last has a next token ("1", "2", ...)."""
    replies = [
        (
            Obj(
                content=[Obj(name=n, type=t) for n, t in page],
                next=str(i + 1) if i + 1 < len(pages) else None,
            ),
            None,
        )
        for i, page in enumerate(pages)
    ]
    return Mock(side_effect=replies)
