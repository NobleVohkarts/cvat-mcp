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
        dimension="2d",
        get_frame=lambda n, quality=None: io.BytesIO(PNG),
        **task_kw,
    )
    api = _echo_api()
    client = fake_client(task, api_client=Obj(tasks_api=api), logger=Mock(), close=Mock())
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


def test_annotate_saves_per_chunk():
    client, task, api = _setup()
    progress = aa.Progress()
    aa.annotate(client, Scope(task), LABELS, FakeFunction(), _opts(), [0, 1, 2, 3, 4], 2, progress)
    assert [sorted({s.frame for s in r.shapes}) for r in _sent(api)] == [[0, 1], [2, 3], [4]]
    assert progress.saved_frames == [0, 1, 2, 3, 4]
    assert progress.added == {"car": 5}
    assert progress.skipped == {"bench": 5}
    assert [o.id for o in progress.objects] == [100, 101, 102, 103, 104]
    assert {(o.label, o.type, o.kind) for o in progress.objects} == {("car", "rectangle", "shape")}
    assert progress.objects[2].frame == 2
    assert progress.objects[0].bbox == [10, 10, 50, 40]
    assert all(s.source == "auto" for r in _sent(api) for s in r.shapes)


def test_annotate_passes_threshold_and_polygons():
    client, task, _ = _setup()
    fn = FakeFunction()
    aa.annotate(client, Scope(task), LABELS, fn, _opts(conf_threshold=0.7), [0], 10, aa.Progress())
    (context,) = fn.contexts
    assert (context.conf_threshold, context.conv_mask_to_poly) == (0.7, True)


def test_annotate_failure_keeps_saved_chunks():
    client, task, api = _setup()
    progress = aa.Progress()
    with pytest.raises(RuntimeError, match="model crashed"):
        aa.annotate(
            client, Scope(task), LABELS, FakeFunction(fail_on=3), _opts(), [0, 1, 2, 3], 2, progress
        )
    assert progress.saved_frames == [0, 1]
    assert len(_sent(api)) == 1


def test_label_type_mismatch_is_invalid_argument():
    client, task, _ = _setup()
    fn = FakeFunction()
    fn.spec = cvataa.DetectionFunctionSpec(
        labels=[cvataa.skeleton_label_spec("car", 0, [cvataa.keypoint_spec("nose", 0)])]
    )
    with pytest.raises(InvalidArgument, match="'car'"):
        aa.annotate(client, Scope(task), LABELS, fn, _opts(), [0], 10, aa.Progress())


def test_replace_deletes_shapes_and_tags_not_tracks():
    client, task, api = _setup(
        shapes=[
            shape(1, 1, 1, [0, 0, 1, 1]),
            shape(2, 1, 2, [0, 0, 1, 1]),
            shape(3, 1, 3, [0, 0, 1, 1]),
        ],
        tags=[tag(4, 2, 2), tag(5, 2, 3)],
        tracks=[Obj(id=9, frame=1, label_id=1, shapes=[])],
    )
    aa.annotate(
        client,
        Scope(task),
        LABELS,
        FakeFunction(),
        _opts(mode="replace"),
        [1, 2],
        10,
        aa.Progress(),
    )
    (deleted,) = _sent(api, "delete")
    assert sorted(s.id for s in deleted.shapes) == [1, 2]
    assert [t.id for t in deleted.tags] == [4]
    assert deleted.tracks == []


def test_replace_with_nothing_there_deletes_nothing():
    client, task, api = _setup()
    aa.annotate(
        client, Scope(task), LABELS, FakeFunction(), _opts(mode="replace"), [0], 10, aa.Progress()
    )
    assert _sent(api, "delete") == []


def test_replace_clears_chunk_by_chunk():
    # A crash in chunk 2 must leave frames 2.. with their old annotations.
    client, task, api = _setup(shapes=[shape(i + 1, 1, i, [0, 0, 1, 1]) for i in range(5)])
    with pytest.raises(RuntimeError):
        aa.annotate(
            client,
            Scope(task),
            LABELS,
            FakeFunction(fail_on=3),
            _opts(mode="replace"),
            [0, 1, 2, 3, 4],
            2,
            aa.Progress(),
        )
    (deleted,) = _sent(api, "delete")
    assert sorted(s.id for s in deleted.shapes) == [1, 2]


def test_replace_with_mismatched_labels_deletes_nothing(use_function):
    client, _, api = _setup(shapes=[shape(1, 1, 0, [0, 0, 1, 1])])
    fn = use_function(FakeFunction())
    fn.spec = cvataa.DetectionFunctionSpec(
        labels=[cvataa.skeleton_label_spec("car", 0, [cvataa.keypoint_spec("nose", 0)])]
    )
    with pytest.raises(InvalidArgument, match="'car'"):
        aa.auto_annotate_frames(client, 1, None, [0], _opts(mode="replace"), 10)
    assert _sent(api, "delete") == []


class UnknownLabel(FakeFunction):
    def detect(self, context, image):
        return [cvataa.rectangle(7, [0, 0, 5, 5])]


def test_output_with_unknown_label_id_is_invalid_argument():
    client, task, _ = _setup()
    with pytest.raises(InvalidArgument, match="unknown"):
        aa.annotate(client, Scope(task), LABELS, UnknownLabel(), _opts(), [0], 10, aa.Progress())


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


def test_target_frames_all_skips_deleted():
    _, task, _ = _setup(deleted_frames=[1])
    assert aa.target_frames(Scope(task), None) == [0, 2, 3, 4]


@pytest.mark.parametrize(
    "frames, match",
    [
        (list(range(21)), "at most 20"),
        ([], "at most 20"),
        ([2, 2], "twice"),
        ([1], "deleted"),
        ([9], "frame 9"),
    ],
)
def test_target_frames_refused(frames, match):
    _, task, _ = _setup(deleted_frames=[1], size=40 if len(frames) > 20 else 5)
    with pytest.raises(InvalidArgument, match=match):
        aa.target_frames(Scope(task), frames)


@pytest.mark.skipif(find_spec("torchvision") is None, reason="needs the auto-annotation extra")
def test_unknown_model_lists_choices():
    with pytest.raises(InvalidArgument, match="maskrcnn_resnet50_fpn_v2"):
        aa.load_function(_opts(function="instance_segmentation", model="nope"))


class BenchOnly(FakeFunction):
    def detect(self, context, image):
        return [cvataa.rectangle(1, [0, 0, 5, 5])]


@pytest.fixture
def use_function(monkeypatch):
    def use(fn):
        monkeypatch.setattr(aa, "load_function", lambda _opts: fn)
        return fn

    return use


def test_frame_run_returns_objects(use_function):
    use_function(FakeFunction())
    client, _, _ = _setup()
    result = aa.auto_annotate_frames(client, 1, None, [1, 0], _opts(), 10)
    assert result.frames == [0, 1]
    assert [(o.id, o.frame, o.label) for o in result.objects] == [(100, 0, "car"), (101, 1, "car")]
    assert result.skipped == {"bench": 2}
    assert (result.task_id, result.job_id, result.mode, result.run) == (1, None, "append", None)
    assert result.model == "fasterrcnn_resnet50_fpn_v2"
    assert result.note is None


def test_frame_run_replace_clears_only_those_frames(use_function):
    use_function(FakeFunction())
    client, _, api = _setup(
        shapes=[
            shape(1, 1, 0, [0, 0, 1, 1]),
            shape(2, 1, 1, [0, 0, 1, 1]),
            shape(3, 1, 2, [0, 0, 1, 1]),
        ]
    )
    aa.auto_annotate_frames(client, 1, None, [0, 1], _opts(mode="replace"), 10)
    (deleted,) = _sent(api, "delete")
    assert sorted(s.id for s in deleted.shapes) == [1, 2]
    actions = [c.args[0] for c in api.partial_update_annotations.call_args_list]
    assert actions == ["delete", "create"]


def test_frame_run_busy_is_refused(use_function):
    use_function(FakeFunction())
    client, _, api = _setup()
    assert aa._LOCK.acquire(blocking=False)
    try:
        with pytest.raises(InvalidArgument, match="in progress"):
            aa.auto_annotate_frames(client, 1, None, [0], _opts(), 10)
    finally:
        aa._LOCK.release()
    api.partial_update_annotations.assert_not_called()


def test_frame_run_partial_failure_lists_saved(use_function):
    use_function(FakeFunction(fail_on=3))
    client, _, _ = _setup()
    with pytest.raises(InvalidArgument) as error:
        aa.auto_annotate_frames(client, 1, None, [0, 1, 2, 3], _opts(), 2)
    message = str(error.value)
    assert "frames 0, 1" in message
    assert "100, 101" in message
    assert "do not repeat" in message
    assert "model crashed" in message
    assert not aa._LOCK.locked()


def test_frame_run_lock_released_after_error(use_function):
    use_function(FakeFunction(fail_on=1))
    client, _, _ = _setup()
    with pytest.raises(RuntimeError):
        aa.auto_annotate_frames(client, 1, None, [0], _opts(), 10)
    assert not aa._LOCK.locked()


def test_frame_run_nothing_matched_has_note(use_function):
    use_function(BenchOnly())
    client, _, _ = _setup()
    result = aa.auto_annotate_frames(client, 1, None, [0], _opts(), 10)
    assert result.objects == []
    assert result.skipped == {"bench": 1}
    assert "car, person" in result.note


@pytest.mark.parametrize("task_id, job_id", [(None, None), (1, 30)])
def test_exactly_one_target(task_id, job_id):
    client, _, _ = _setup()
    with pytest.raises(InvalidArgument, match="exactly one"):
        aa.auto_annotate_frames(client, task_id, job_id, [0], _opts(), 10)


def _start(fn, use_function, **setup_kw):
    """Start a background run whose thread uses its own fake client; returns what tests check."""
    use_function(fn)
    request_client, _, request_api = _setup()
    run_client, _, run_api = _setup(**setup_kw)
    result = aa.start_run(request_client, lambda: run_client, 1, None, _opts(), 2)
    return result, request_api, run_client, run_api


def _finish(run_id):
    for thread in threading.enumerate():
        if thread.name == f"auto-annotation {run_id}":
            thread.join(timeout=10)
    (run,) = aa.get_runs(run_id).runs
    return run


def test_background_run_finishes(use_function):
    result, _, _, _ = _start(FakeFunction(), use_function)
    assert result.run.state == "running"
    assert (result.run.frames_total, result.objects) == (5, [])
    run = _finish(result.run.run_id)
    assert (run.state, run.frames_done, run.added, run.skipped) == (
        "finished",
        5,
        {"car": 5},
        {"bench": 5},
    )
    assert run.finished is not None and run.error is None
    assert aa._RUNS[run.run_id][1].objects == []  # background runs keep no object list
    assert not aa._LOCK.locked()


def test_background_run_failure_recorded_and_lock_released(use_function):
    result, _, run_client, _ = _start(FakeFunction(fail_on=3), use_function)
    run = _finish(result.run.run_id)
    assert (run.state, run.frames_done) == ("failed", 2)
    assert "model crashed" in run.error
    assert not aa._LOCK.locked()
    run_client.close.assert_called_once()


def test_background_run_uses_own_client(use_function):
    result, request_api, run_client, run_api = _start(FakeFunction(), use_function)
    _finish(result.run.run_id)
    request_api.partial_update_annotations.assert_not_called()
    assert len(_sent(run_api)) == 3  # 5 frames in chunks of 2
    run_client.close.assert_called_once()


def test_background_replace_clears_all_target_frames(use_function):
    use_function(FakeFunction())
    request_client, _, _ = _setup()
    run_client, _, run_api = _setup(
        shapes=[shape(1, 1, 0, [0, 0, 1, 1]), shape(2, 1, 4, [0, 0, 1, 1])],
        tracks=[Obj(id=9, frame=1, label_id=1, shapes=[])],
    )
    result = aa.start_run(request_client, lambda: run_client, 1, None, _opts(mode="replace"), 2)
    _finish(result.run.run_id)
    deletes = _sent(run_api, "delete")  # one per chunk that had old objects: frames 0 and 4
    assert [[s.id for s in d.shapes] for d in deletes] == [[1], [2]]
    assert all(d.tracks == [] for d in deletes)


def test_get_runs_unknown_id():
    with pytest.raises(InvalidArgument, match="unknown run_id"):
        aa.get_runs("run-0")


def test_get_runs_lists_newest_first(use_function):
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


def test_call_while_background_run_names_run_id(use_function):
    fn = Blocking()
    result, _, _, _ = _start(fn, use_function)
    run_id = result.run.run_id
    try:
        client, _, _ = _setup()
        with pytest.raises(InvalidArgument, match=f"background run {run_id} is in progress"):
            aa.auto_annotate_frames(client, 1, None, [0], _opts(), 10)
    finally:
        fn.go.set()
    assert _finish(run_id).state == "finished"
