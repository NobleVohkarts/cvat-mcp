import io
import itertools
from importlib.util import find_spec
from unittest.mock import Mock

import cvat_sdk.auto_annotation as cvataa
import PIL.Image
import pytest
from fakes import Obj, fake_client, fake_task, shape, tag

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services import auto_annotation as aa
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
        size=size, labels=LABELS, get_frame=lambda n, quality=None: io.BytesIO(PNG), **task_kw
    )
    api = _echo_api()
    client = fake_client(task, api_client=Obj(tasks_api=api), logger=Mock())
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
    assert progress.frames_done == 5
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


def test_clear_frames_deletes_shapes_and_tags_not_tracks():
    task = fake_task(
        labels=LABELS,
        shapes=[
            shape(1, 1, 1, [0, 0, 1, 1]),
            shape(2, 1, 2, [0, 0, 1, 1]),
            shape(3, 1, 3, [0, 0, 1, 1]),
        ],
        tags=[tag(4, 2, 2), tag(5, 2, 3)],
        tracks=[Obj(id=9, frame=1, label_id=1, shapes=[])],
    )
    api = _echo_api()
    client = fake_client(task, api_client=Obj(tasks_api=api))
    aa.clear_frames(client, Scope(task), [1, 2])
    (deleted,) = _sent(api, "delete")
    assert sorted(s.id for s in deleted.shapes) == [1, 2]
    assert [t.id for t in deleted.tags] == [4]
    assert deleted.tracks == []


def test_clear_frames_with_nothing_there_sends_nothing():
    client, task, api = _setup()
    aa.clear_frames(client, Scope(task), [0])
    api.partial_update_annotations.assert_not_called()


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
