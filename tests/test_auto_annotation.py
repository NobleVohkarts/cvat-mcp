import io
import itertools
import threading
from importlib.util import find_spec
from unittest.mock import Mock

import cvat_sdk.auto_annotation as cvataa
import PIL.Image
import pytest
from fakes import Obj, fake_client, fake_task, shape, tag

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services import auto_annotation as aa
from cvat_mcp.services.annotation_edits import create_objects
from cvat_mcp.services.annotations import Scope
from cvat_mcp.tools import auto_annotation as tools

LABELS = [
    Obj(id=1, name="car", type="rectangle", attributes=[], sublabels=[]),
    Obj(id=2, name="person", type="any", attributes=[], sublabels=[]),
]


def _png() -> bytes:
    buffer = io.BytesIO()
    PIL.Image.new("RGB", (64, 48)).save(buffer, format="PNG")
    return buffer.getvalue()


PNG = _png()


class FakeFunction:
    """Outputs a car and a bench on every frame; the task has no bench label."""

    def __init__(self, fail_on: int | None = None):
        self.spec = cvataa.DetectionFunctionSpec(
            labels=[
                cvataa.label_spec("car", 0, type="rectangle"),
                cvataa.label_spec("bench", 1, type="rectangle"),
            ]
        )
        self.contexts = []
        self.fail_on = fail_on

    def detect(self, context, image):
        self.contexts.append(context)
        if len(self.contexts) == self.fail_on:
            raise RuntimeError("model crashed")
        return [cvataa.rectangle(0, [10, 10, 50, 40]), cvataa.rectangle(1, [0, 0, 5, 5])]


class BenchOnly(FakeFunction):
    def detect(self, context, image):
        return [cvataa.rectangle(1, [0, 0, 5, 5])]


class UnknownLabel(FakeFunction):
    def detect(self, context, image):
        return [cvataa.rectangle(7, [0, 0, 5, 5])]


def _skeleton_car(fn):
    fn.spec = cvataa.DetectionFunctionSpec(
        labels=[cvataa.skeleton_label_spec("car", 0, [cvataa.keypoint_spec("nose", 0)])]
    )
    return fn


def _echo_api():
    """tasks_api whose create PATCH answers with the sent objects plus new ids."""
    api = Mock()
    ids = itertools.count(100)

    def patch(action, _id, patched_labeled_data_request):
        sent = patched_labeled_data_request
        if action != "create":
            return None, None
        shapes = [
            Obj(
                id=next(ids),
                frame=s.frame,
                label_id=s.label_id,
                type=s.type,
                points=s.points,
                occluded=False,
            )
            for s in sent.shapes
        ]
        tags = [Obj(id=next(ids), frame=t.frame, label_id=t.label_id) for t in sent.tags]
        return Obj(shapes=shapes, tags=tags), None

    api.partial_update_annotations.side_effect = patch
    return api


def _setup(size=5, **task_kw):
    task = fake_task(
        size=size,
        labels=LABELS,
        get_frame=lambda n, quality=None: io.BytesIO(PNG),
        **{"dimension": "2d", "data_original_chunk_type": "imageset"} | task_kw,
    )
    api = _echo_api()
    client = fake_client(
        task,
        api_client=Obj(tasks_api=api),
        logger=Mock(),
        close=Mock(),
        config=Obj(cache_dir=None),
    )
    return client, task, api


def _opts(**kw):
    return aa.check_options(
        **(
            {
                "function": "detection",
                "model": None,
                "conf_threshold": 0.5,
                "polygons": True,
                "mode": "append",
            }
            | kw
        )
    )


def _sent(api, action="create"):
    return [
        c.kwargs["patched_labeled_data_request"]
        for c in api.partial_update_annotations.call_args_list
        if c.args[0] == action
    ]


@pytest.fixture
def use_function(monkeypatch):
    def use(fn):
        monkeypatch.setattr(aa, "load_function", lambda _opts: fn)
        return fn

    return use


# --- label matching ---


POSE = Obj(
    id=5,
    name="pose",
    type="skeleton",
    attributes=[],
    sublabels=[Obj(id=20, name="nose"), Obj(id=21, name="eye")],
)


def _pose_function(*points):
    fn = FakeFunction()
    fn.spec = cvataa.DetectionFunctionSpec(
        labels=[
            cvataa.skeleton_label_spec(
                "pose", 0, [cvataa.keypoint_spec(name, i) for i, name in enumerate(points)]
            )
        ]
    )
    return fn


def test_label_map_skeleton_points():
    mapping = aa.label_map(_pose_function("nose", "eye", "ear"), [POSE], polygons=True)
    assert mapping == {0: (5, {0: 20, 1: 21})}
    skeleton = cvataa.skeleton(
        0,
        [cvataa.keypoint(0, [1, 2]), cvataa.keypoint(1, [3, 4]), cvataa.keypoint(2, [5, 6])],
    )
    ((kind, remapped),) = aa.remap([skeleton], mapping, 7)
    assert (kind, remapped.label_id, remapped.frame, remapped.source) == ("shape", 5, 7, "auto")
    assert [(e.label_id, e.frame, e.source) for e in remapped.elements] == [
        (20, 7, "auto"),
        (21, 7, "auto"),
    ]


def test_label_map_missing_points_refused():
    tail = Obj(
        id=5,
        name="pose",
        type="skeleton",
        sublabels=[Obj(id=20, name="nose"), Obj(id=22, name="tail")],
    )
    with pytest.raises(InvalidArgument, match="tail"):
        aa.label_map(_pose_function("nose", "eye"), [tail], polygons=True)


def test_label_map_skeleton_into_box_refused():
    with pytest.raises(InvalidArgument, match="'car'"):
        aa.label_map(_skeleton_car(FakeFunction()), LABELS, polygons=True)


def test_label_map_mask_as_polygon():
    fn = FakeFunction()
    fn.spec = cvataa.DetectionFunctionSpec(labels=[cvataa.label_spec("blob", 0, type="mask")])
    blob = Obj(id=3, name="blob", type="polygon", sublabels=[])
    assert aa.label_map(fn, [blob], polygons=True) == {0: (3, {})}
    with pytest.raises(InvalidArgument, match="'blob'"):
        aa.label_map(fn, [blob], polygons=False)


def test_label_map_any_takes_everything_and_missing_is_none():
    mapping = aa.label_map(FakeFunction(), [Obj(id=9, name="car", type="any", sublabels=[])], True)
    assert mapping == {0: (9, {}), 1: None}


# --- one frame ---


def test_frame_returns_objects(use_function):
    use_function(FakeFunction())
    client, _, api = _setup()
    result = aa.auto_annotate_frame(client, 1, 2, _opts())
    assert result.frame == 2
    assert [(o.id, o.frame, o.label) for o in result.objects] == [(100, 2, "car")]
    assert result.objects[0].bbox == [10, 10, 50, 40]
    assert result.skipped == {"bench": 1}
    assert (result.task_id, result.mode, result.run) == (1, "append", None)
    assert result.model == "fasterrcnn_resnet50_fpn_v2"
    assert result.note is None
    assert all(s.source == "auto" for r in _sent(api) for s in r.shapes)


def test_frame_passes_threshold_and_polygons(use_function):
    fn = use_function(FakeFunction())
    client, _, _ = _setup()
    aa.auto_annotate_frame(client, 1, 0, _opts(conf_threshold=0.7))
    (context,) = fn.contexts
    assert isinstance(context, cvataa.DetectionFunctionContext)
    assert (context.conf_threshold, context.conv_mask_to_poly) == (0.7, True)


def test_frame_replace_clears_only_that_frame(use_function):
    use_function(FakeFunction())
    client, _, api = _setup(
        shapes=[shape(1, 1, 0, [0, 0, 1, 1]), shape(2, 1, 1, [0, 0, 1, 1])],
        tags=[tag(4, 2, 1)],
        tracks=[Obj(id=9, frame=1, label_id=1, shapes=[])],
    )
    aa.auto_annotate_frame(client, 1, 1, _opts(mode="replace"))
    (deleted,) = _sent(api, "delete")
    assert [s.id for s in deleted.shapes] == [2]
    assert [t.id for t in deleted.tags] == [4]
    assert deleted.tracks == []
    # new first: a failed create must leave the old ones
    assert [c.args[0] for c in api.partial_update_annotations.call_args_list] == [
        "create",
        "delete",
    ]


def _failing_delete(api):
    create = api.partial_update_annotations.side_effect

    def patch(action, _id, patched_labeled_data_request):
        if action == "delete":
            raise RuntimeError("CVAT down")
        return create(action, _id, patched_labeled_data_request)

    api.partial_update_annotations.side_effect = patch


def test_frame_replace_delete_failure_says_new_ones_are_saved(use_function):
    use_function(FakeFunction())
    client, _, api = _setup(shapes=[shape(1, 1, 0, [0, 0, 1, 1])])
    _failing_delete(api)
    with pytest.raises(InvalidArgument) as error:
        aa.auto_annotate_frame(client, 1, 0, _opts(mode="replace"))
    message = str(error.value)
    assert "saved" in message and "100" in message and "Do not repeat" in message
    assert "CVAT down" in message
    assert not aa._LOCK.locked()


def test_frame_replace_create_failure_deletes_nothing(use_function):
    use_function(FakeFunction())
    client, _, api = _setup(shapes=[shape(1, 1, 0, [0, 0, 1, 1])])
    api.partial_update_annotations.side_effect = RuntimeError("CVAT down")
    with pytest.raises(RuntimeError, match="CVAT down"):
        aa.auto_annotate_frame(client, 1, 0, _opts(mode="replace"))
    assert [c.args[0] for c in api.partial_update_annotations.call_args_list] == ["create"]


def test_frame_replace_with_nothing_there_deletes_nothing(use_function):
    use_function(FakeFunction())
    client, _, api = _setup()
    aa.auto_annotate_frame(client, 1, 0, _opts(mode="replace"))
    assert _sent(api, "delete") == []


def test_frame_replace_with_mismatched_labels_deletes_nothing(use_function):
    use_function(_skeleton_car(FakeFunction()))
    client, _, api = _setup(shapes=[shape(1, 1, 0, [0, 0, 1, 1])])
    with pytest.raises(InvalidArgument, match="'car'"):
        aa.auto_annotate_frame(client, 1, 0, _opts(mode="replace"))
    api.partial_update_annotations.assert_not_called()


def test_frame_unknown_label_id_is_invalid_argument(use_function):
    use_function(UnknownLabel())
    client, _, api = _setup()
    with pytest.raises(InvalidArgument, match="unknown"):
        aa.auto_annotate_frame(client, 1, 0, _opts())
    api.partial_update_annotations.assert_not_called()


@pytest.mark.parametrize("frame, match", [(1, "frame 1 is deleted"), (9, "frame 9")])
def test_frame_refused(use_function, frame, match):
    use_function(FakeFunction())
    client, _, _ = _setup(deleted_frames=[1])
    with pytest.raises(InvalidArgument, match=match):
        aa.auto_annotate_frame(client, 1, frame, _opts())


def test_frame_refuses_3d(use_function):
    use_function(FakeFunction())
    client, _, _ = _setup(dimension="3d")
    with pytest.raises(InvalidArgument, match="3D"):
        aa.auto_annotate_frame(client, 1, 0, _opts())


def test_frame_busy_is_refused(use_function):
    use_function(FakeFunction())
    client, _, api = _setup()
    assert aa._LOCK.acquire(blocking=False)
    try:
        with pytest.raises(InvalidArgument, match="in progress"):
            aa.auto_annotate_frame(client, 1, 0, _opts())
    finally:
        aa._LOCK.release()
    api.partial_update_annotations.assert_not_called()


def test_frame_lock_released_after_error(use_function):
    use_function(FakeFunction(fail_on=1))
    client, _, _ = _setup()
    with pytest.raises(RuntimeError):
        aa.auto_annotate_frame(client, 1, 0, _opts())
    assert not aa._LOCK.locked()


def test_frame_nothing_matched_has_note(use_function):
    use_function(BenchOnly())
    client, _, _ = _setup()
    result = aa.auto_annotate_frame(client, 1, 0, _opts())
    assert result.objects == []
    assert result.skipped == {"bench": 1}
    assert "car, person" in result.note


# --- whole task in the background ---


class FakeAnnotateTask:
    """Stands in for cvat_sdk.auto_annotation.annotate_task: runs the function on every frame."""

    def __init__(self):
        self.calls = []
        self.fail_after: int | None = None
        self.cache_dirs = []

    def __call__(self, client, task_id, function, *, pbar, **kw):
        self.calls.append((client, task_id, kw))
        self.cache_dirs.append(client.config.cache_dir)
        (client.config.cache_dir / "chunk.zip").write_bytes(PNG)  # what TaskDataset leaves
        image = PIL.Image.open(io.BytesIO(PNG))
        size = client.tasks.retrieve(task_id).size
        with pbar.task(total=size, unit="samples"):
            for i in pbar.iter(range(size)):
                if i == self.fail_after:
                    raise RuntimeError("model crashed")
                function.detect(
                    aa.Context(str(i), kw["conf_threshold"], kw["conv_mask_to_poly"]), image
                )


@pytest.fixture
def annotate_task(monkeypatch):
    fake = FakeAnnotateTask()
    monkeypatch.setattr(aa, "annotate_task", fake)
    return fake


def _start(fn, use_function, opts=None, **setup_kw):
    """Start a background run whose thread uses its own fake client."""
    use_function(fn)
    request_client, _, request_api = _setup()
    run_client, _, run_api = _setup(**setup_kw)
    result = aa.start_run(request_client, lambda: run_client, 1, opts or _opts())
    return result, request_api, run_client, run_api


def _finish(run_id):
    for thread in threading.enumerate():
        if thread.name == f"auto-annotation {run_id}":
            thread.join(timeout=10)
    (run,) = aa.get_runs(run_id).runs
    return run


def test_background_run_finishes(use_function, annotate_task):
    result, _, _, _ = _start(FakeFunction(), use_function)
    assert result.run.state == "running"
    assert (result.run.frames_total, result.objects, result.frame) == (5, [], None)
    run = _finish(result.run.run_id)
    assert (run.state, run.frames_done, run.added, run.skipped) == (
        "finished",
        5,
        {"car": 5},
        {"bench": 5},
    )
    assert run.finished is not None and run.error is None
    ((_, task_id, kw),) = annotate_task.calls
    assert task_id == 1
    assert kw == {
        "clear_existing": False,
        "allow_unmatched_labels": True,
        "conf_threshold": 0.5,
        "conv_mask_to_poly": True,
    }
    assert not aa._LOCK.locked()


def test_background_run_failure_recorded_and_lock_released(use_function, annotate_task):
    annotate_task.fail_after = 2
    result, _, run_client, _ = _start(FakeFunction(), use_function)
    run = _finish(result.run.run_id)
    assert (run.state, run.frames_done) == ("failed", 2)
    assert "model crashed" in run.error
    assert not aa._LOCK.locked()
    run_client.close.assert_called_once()


@pytest.mark.parametrize("fail_after", [None, 1])
def test_background_run_removes_downloaded_images(use_function, annotate_task, fail_after):
    annotate_task.fail_after = fail_after
    result, _, _, _ = _start(FakeFunction(), use_function)
    _finish(result.run.run_id)
    (cache_dir,) = annotate_task.cache_dirs
    assert not cache_dir.exists()


def test_background_run_uses_own_client(use_function, annotate_task):
    result, request_api, run_client, _ = _start(FakeFunction(), use_function)
    _finish(result.run.run_id)
    ((client, _, _),) = annotate_task.calls
    assert client is run_client
    request_api.partial_update_annotations.assert_not_called()
    run_client.close.assert_called_once()


def test_background_replace_deletes_old_after_upload(use_function, annotate_task):
    result, _, _, run_api = _start(
        FakeFunction(),
        use_function,
        _opts(mode="replace"),
        shapes=[shape(1, 1, 0, [0, 0, 1, 1]), shape(2, 1, 4, [0, 0, 1, 1])],
        tags=[tag(3, 2, 2)],
        tracks=[Obj(id=9, frame=1, label_id=1, shapes=[])],
    )
    assert _finish(result.run.run_id).state == "finished"
    (deleted,) = _sent(run_api, "delete")
    assert sorted(s.id for s in deleted.shapes) == [1, 2]
    assert [t.id for t in deleted.tags] == [3]
    assert deleted.tracks == []


def test_background_replace_delete_failure_says_new_ones_are_saved(use_function, annotate_task):
    use_function(FakeFunction())
    request_client, _, _ = _setup()
    run_client, _, run_api = _setup(shapes=[shape(1, 1, 0, [0, 0, 1, 1])])
    _failing_delete(run_api)
    result = aa.start_run(request_client, lambda: run_client, 1, _opts(mode="replace"))
    run = _finish(result.run.run_id)
    assert run.state == "failed"
    assert "saved" in run.error and "Do not repeat" in run.error and "CVAT down" in run.error


def test_background_run_setup_failure_releases_lock(use_function, annotate_task, monkeypatch):
    def no_space(**_):
        raise OSError("No space left on device")

    monkeypatch.setattr(aa.tempfile, "TemporaryDirectory", no_space)
    result, _, _, _ = _start(FakeFunction(), use_function)
    run = _finish(result.run.run_id)
    assert run.state == "failed" and "No space left" in run.error
    assert not aa._LOCK.locked()


def test_background_replace_failure_deletes_nothing(use_function, annotate_task):
    annotate_task.fail_after = 3
    result, _, _, run_api = _start(
        FakeFunction(), use_function, _opts(mode="replace"), shapes=[shape(1, 1, 0, [0, 0, 1, 1])]
    )
    assert _finish(result.run.run_id).state == "failed"
    run_api.partial_update_annotations.assert_not_called()


def test_background_refuses_video(use_function, annotate_task):
    use_function(FakeFunction())
    client, _, _ = _setup(data_original_chunk_type="video")
    before = len(aa.get_runs(None).runs)
    with pytest.raises(InvalidArgument, match="pass frame"):
        aa.start_run(client, lambda: client, 1, _opts())
    assert len(aa.get_runs(None).runs) == before
    assert not aa._LOCK.locked()
    assert annotate_task.calls == []


def test_background_label_mismatch_fails_call(use_function, annotate_task):
    use_function(_skeleton_car(FakeFunction()))
    client, _, _ = _setup()
    before = len(aa.get_runs(None).runs)
    with pytest.raises(InvalidArgument, match="'car'"):
        aa.start_run(client, lambda: client, 1, _opts())
    assert len(aa.get_runs(None).runs) == before
    assert not aa._LOCK.locked()


def test_get_runs_unknown_id():
    with pytest.raises(InvalidArgument, match="unknown run_id"):
        aa.get_runs("run-0")


def test_get_runs_lists_newest_first(use_function, annotate_task):
    first, _, _, _ = _start(FakeFunction(), use_function)
    _finish(first.run.run_id)
    second, _, _, _ = _start(FakeFunction(), use_function)
    _finish(second.run.run_id)
    ids = [r.run_id for r in aa.get_runs(None).runs]
    assert ids.index(second.run.run_id) < ids.index(first.run.run_id)
    assert "restart" in aa.get_runs(None).note


class Blocking(FakeFunction):
    def __init__(self):
        super().__init__()
        self.go = threading.Event()

    def detect(self, context, image):
        self.go.wait(timeout=10)
        return super().detect(context, image)


def test_call_while_background_run_names_run_id(use_function, annotate_task):
    fn = Blocking()
    result, _, _, _ = _start(fn, use_function)
    run_id = result.run.run_id
    try:
        client, _, _ = _setup()
        with pytest.raises(InvalidArgument, match=f"background run {run_id} is in progress"):
            aa.auto_annotate_frame(client, 1, 0, _opts())
    finally:
        fn.go.set()
    assert _finish(run_id).state == "finished"


# --- unchanged helpers ---


def test_create_objects_pairs_skeletons():
    # Skeleton requests have no points of their own; the key must come from the elements.
    skeleton = cvataa.skeleton(1, [cvataa.keypoint(0, [5, 6])])
    skeleton.frame = 0
    answer = Obj(
        shapes=[
            Obj(
                id=55,
                frame=0,
                label_id=1,
                type=skeleton.type,
                points=[],
                occluded=False,
                elements=[Obj(points=[5.0, 6.0])],
            )
        ],
        tags=[],
    )
    api = Mock()
    api.partial_update_annotations.return_value = (answer, None)
    task = fake_task(labels=LABELS)
    client = fake_client(task, api_client=Obj(tasks_api=api))
    assert create_objects(client, Scope(task), [("shape", skeleton)]) == [55]


def test_create_objects_never_fails_after_the_write():
    api = Mock()
    api.partial_update_annotations.return_value = (Obj(shapes=[Obj(id=1)], tags=[]), None)
    task = fake_task(labels=LABELS)
    client = fake_client(task, api_client=Obj(tasks_api=api))
    sent = [("shape", cvataa.rectangle(1, [0, 0, 1, 1]))]
    assert create_objects(client, Scope(task), sent) == [None]


def test_check_options_threshold_out_of_range():
    with pytest.raises(InvalidArgument, match="conf_threshold"):
        _opts(conf_threshold=1.5)


@pytest.mark.parametrize("function, model", list(aa.DEFAULT_MODELS.items()))
def test_check_options_default_model(function, model):
    assert _opts(function=function).model == model


@pytest.mark.skipif(find_spec("torchvision") is None, reason="needs the auto-annotation extra")
def test_unknown_model_lists_choices():
    with pytest.raises(InvalidArgument, match="maskrcnn_resnet50_fpn_v2"):
        aa.load_function(_opts(function="instance_segmentation", model="nope"))


# --- the tool layer ---


def test_tool_routes_whole_task_and_one_frame(monkeypatch):
    calls = []
    monkeypatch.setattr(aa, "start_run", lambda *args: calls.append(("run", args)) or "run")
    monkeypatch.setattr(
        aa, "auto_annotate_frame", lambda *args: calls.append(("frame", args)) or "frame"
    )
    client = object()
    ctx = Obj(request_context=Obj(lifespan_context=client))
    assert tools.auto_annotate(ctx, task_id=7) == "run"
    assert tools.auto_annotate(ctx, task_id=7, frame=3, mode="replace") == "frame"
    assert calls == [
        ("run", (client, tools.open_client, 7, _opts())),
        ("frame", (client, 7, 3, _opts(mode="replace"))),
    ]
