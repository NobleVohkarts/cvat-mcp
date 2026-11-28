import io
from unittest.mock import Mock

import pytest
from fakes import fake_client, fake_task, gt_job, job_client, shape
from PIL import Image

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.render import (
    _overlaps,
    annotate,
    clamp_side,
    clip_region,
    legend,
    place_caption,
    render,
    render_frame,
    resize,
)
from cvat_mcp.services.annotations import FrameObject, shape_bbox

CAR = FrameObject(
    id=1,
    label="car",
    type="rectangle",
    bbox=[100, 100, 300, 250],
    attributes={},
    source="manual",
    occluded=False,
)


def _red(pixel):
    return pixel[0] > 200 and pixel[1] < 80


def _black():
    return Image.new("RGB", (800, 600))


def test_downscales_to_max_side():
    out, scale = render(_black(), [CAR], {"car": "#ff0000"}, 400)
    assert out.size == (400, 300)
    assert scale == 0.5


def test_box_drawn_at_scaled_position():
    out, _ = render(_black(), [CAR], {"car": "#ff0000"}, 400)
    assert _red(out.getpixel((50, 100)))  # left edge, mid-height of the scaled box
    assert out.getpixel((100, 87)) == (0, 0, 0)  # inside the box stays untouched


def _shape(type_, points):
    bbox = shape_bbox(shape(1, 1, 0, points, type=type_))
    return CAR.model_copy(update={"type": type_, "points": points, "bbox": bbox})


TRIANGLE = [100, 100, 300, 100, 100, 300]


def test_polygon_drawn_by_its_points_not_its_box():
    out, _ = render(_black(), [_shape("polygon", TRIANGLE)], {"car": "#ff0000"}, 800)
    assert _red(out.getpixel((200, 200)))  # on the diagonal edge
    assert _red(out.getpixel((100, 200)))  # the closing edge
    assert out.getpixel((300, 250)) == (0, 0, 0)  # the box's right edge is not drawn


def test_polyline_is_left_open():
    out, _ = render(_black(), [_shape("polyline", TRIANGLE)], {"car": "#ff0000"}, 800)
    assert _red(out.getpixel((200, 200)))
    assert out.getpixel((100, 200)) == (0, 0, 0)  # no closing edge


def test_points_drawn_as_dots():
    out, _ = render(_black(), [_shape("points", [400, 400, 500, 450])], {"car": "#ff0000"}, 800)
    assert _red(out.getpixel((400, 400))) and _red(out.getpixel((500, 450)))
    assert out.getpixel((500, 425)) == (0, 0, 0)  # the box's right edge is not drawn


def test_never_upscales():
    out, scale = render(_black(), [CAR], {"car": "#ff0000"}, 2048)
    assert out.size == (800, 600) and scale == 1.0


def test_legend_uses_original_coordinates():
    lines = legend([CAR], 0.5, 7).splitlines()
    assert "frame 7" in lines[0] and "0.500" in lines[0]
    assert lines[1] == "#1 car [100.0, 100.0, 300.0, 250.0]"


def test_legend_empty_frame():
    assert "no objects" in legend([], 1.0, 0)


def test_legend_mentions_tracks_not_drawn():
    text = legend([], 1.0, 0, skipped_tracks=2)
    assert "2 tracked objects are not drawn" in text and "no objects" not in text


def test_caption_stays_inside_image_at_right_edge():
    edge = CAR.model_copy(update={"bbox": [780, 100, 799, 250]})
    out, _ = render(_black(), [edge], {"car": "#ff0000"}, 2048)
    r, g, _b = out.getpixel((770, 95))  # caption bar shifted left of the box
    assert r > 100 and g < 40  # semi-transparent red over black


def test_clamp_side():
    assert clamp_side(10) == 256 and clamp_side(5000) == 2048 and clamp_side(1280) == 1280


def test_resize_zoom_is_capped():
    out, scale = resize(Image.new("RGB", (10, 10)), 1280, max_zoom=4.0)
    assert out.size == (40, 40) and scale == 4.0


def test_resize_zoom_still_fits_max_side():
    out, scale = resize(Image.new("RGB", (400, 200)), 1280, max_zoom=4.0)
    assert out.size == (1280, 640) and scale == 3.2


def test_annotate_offset_moves_the_box():
    out = annotate(Image.new("RGB", (400, 300)), [CAR], {"car": "#ff0000"}, 2.0, (100, 100))
    assert _red(out.getpixel((1, 150)))  # left edge: (100 - 100) * 2 = 0
    assert out.getpixel((200, 150)) == (0, 0, 0)  # inside stays untouched


def _png(size):
    def get_frame(_frame, quality):  # pylint: disable=unused-argument
        buffer = io.BytesIO()
        Image.new("RGB", size).save(buffer, "PNG")
        buffer.seek(0)
        return buffer

    return get_frame


_frame_png = _png((640, 480))


def _render(shapes, include_clean=False, region=None, frame_size=(640, 480)):
    task = fake_task(shapes, get_frame=_png(frame_size))
    return render_frame(fake_client(task), 1, 0, 1280, include_clean=include_clean, region=region)


CAR_SHAPE = shape(1, 1, 0, [100, 100, 300, 250])


def test_region_crops_and_zooms():
    jpegs, text = _render([CAR_SHAPE], region=[100, 100, 260, 220])  # 160x120 -> 4x
    image = Image.open(io.BytesIO(jpegs[0]))
    assert image.size == (640, 480)
    assert image.getpixel((1, 240))[0] > 150  # the box's left edge at x = 0 (JPEG: loose)
    assert "region [100, 100, 260, 220], 1 objects touch it, image scaled by 4.000" in text
    assert "#1 car [100.0, 100.0, 300.0, 250.0]" in text


def test_region_lists_only_touching_objects():
    shapes = [CAR_SHAPE, shape(2, 1, 0, [500, 400, 600, 470])]
    _, text = _render(shapes, region=[0, 0, 320, 240])
    assert "#1 " in text and "#2 " not in text and "1 objects touch it" in text


def test_region_partly_outside_is_clipped():
    _, text = _render([], region=[600, 400, 700, 500])
    assert "region [600, 400, 640, 480]" in text


def test_region_keeps_zero_area_objects():
    dot = shape(1, 1, 0, [150, 150], type="points")
    _, text = _render([dot], region=[100, 100, 200, 200])
    assert "#1 " in text


def test_region_outside_frame_refused_before_download():
    get_frame = Mock(side_effect=_png((640, 480)))
    task = fake_task([CAR_SHAPE], get_frame=get_frame)
    with pytest.raises(InvalidArgument, match="640x480"):
        render_frame(fake_client(task), 1, 0, 1280, region=[700, 0, 800, 100])
    get_frame.assert_not_called()


def test_region_with_include_clean():
    jpegs, _ = _render([CAR_SHAPE], region=[100, 100, 260, 220], include_clean=True)
    annotated, clean = (Image.open(io.BytesIO(j)) for j in jpegs)
    assert annotated.size == clean.size == (640, 480)
    assert clean.getpixel((1, 240))[0] < 40  # nothing drawn on the clean crop


def test_whole_frame_maps_onto_a_smaller_compressed_frame():
    jpegs, text = _render([CAR_SHAPE], frame_size=(320, 240))  # meta: 640x480
    image = Image.open(io.BytesIO(jpegs[0]))
    assert image.size == (320, 240)  # never enlarged without a region
    assert image.getpixel((50, 90))[0] > 150  # box left edge at 100 * 0.5 (was drawn at 100)
    assert "image scaled by 0.500" in text


def test_region_maps_onto_a_smaller_compressed_frame():
    jpegs, text = _render(
        [CAR_SHAPE], region=[0, 0, 320, 240], frame_size=(320, 240)
    )  # meta: 640x480
    assert Image.open(io.BytesIO(jpegs[0])).size == (640, 480)  # crop 160x120, zoom 4
    assert "image scaled by 2.000" in text


@pytest.mark.parametrize(
    "region, clipped",
    [
        ([10, 20, 30, 40], (10, 20, 30, 40)),
        ([-50, -50, 100, 100], (0, 0, 100, 100)),
        ([600, 400, 700, 500], (600, 400, 640, 480)),
    ],
)
def test_clip_region(region, clipped):
    assert clip_region(region, 640, 480) == clipped


@pytest.mark.parametrize(
    "region, match",
    [([700, 0, 800, 100], r"outside the frame \(640x480"), ([30, 30, 10, 10], "x1 < x2")],
)
def test_clip_region_refuses(region, match):
    with pytest.raises(InvalidArgument, match=match):
        clip_region(region, 640, 480)


def test_caption_goes_above_a_free_box():
    assert place_caption((100, 100, 200, 200), (50, 14), (640, 480), []) == (100, 86)


def test_caption_goes_inside_when_the_box_touches_the_top():
    assert place_caption((100, 5, 200, 200), (50, 14), (640, 480), []) == (100, 5)


def test_caption_avoids_taken_places():
    taken = [(100, 86, 150, 100), (100, 100, 150, 114)]
    assert place_caption((100, 100, 200, 200), (50, 14), (640, 480), taken) == (100, 200)


def test_neighbouring_captions_do_not_overlap():
    taken = []
    for box in [(100, 100, 200, 200), (130, 100, 230, 200), (160, 100, 260, 200)]:
        x, y = place_caption(box, (50, 14), (640, 480), taken)
        rect = (x, y, x + 50, y + 14)
        assert not any(_overlaps(rect, t) for t in taken)
        taken.append(rect)


def test_caption_bar_lets_the_image_show_through():
    white = Image.new("RGB", (800, 600), "white")
    out, _ = render(white, [CAR], {"car": "#ff0000"}, 400)
    r, g, b = out.getpixel((50, 47))  # left padding of the caption bar above the scaled box
    assert r > 200 and 60 < g < 230 and 60 < b < 230  # red blended with white, not solid


def test_render_frame_can_add_the_clean_frame():
    jpegs, text = _render([CAR_SHAPE], True)
    annotated, clean = (Image.open(io.BytesIO(j)) for j in jpegs)
    assert annotated.size == clean.size == (640, 480)
    assert annotated.getpixel((100, 175))[0] > 150  # the red box edge
    assert clean.getpixel((100, 175))[0] < 40  # nothing drawn on the clean copy
    assert "without annotations" in text.splitlines()[0]


def test_render_frame_default_is_one_image():
    jpegs, text = _render([CAR_SHAPE], False)
    assert len(jpegs) == 1 and "without annotations" not in text


def test_clean_frame_for_an_empty_frame():
    jpegs, text = _render([], True)
    assert len(jpegs) == 2 and "no objects" in text


def test_caption_text_contrasts_with_the_blended_bar():
    # mid-green label over a white frame: the blended bar is light, so the text must be dark
    white = Image.new("RGB", (800, 600), "white")
    out, _ = render(white, [CAR], {"car": "#00a000"}, 400)
    bar = out.crop((50, 34, 150, 49))  # caption area above the scaled box
    assert min(channel_min for channel_min, _ in bar.getextrema()) < 60


def test_overlapping_fallback_caption_is_opaque():
    # six identical boxes: four caption spots, then captions must overlap
    boxes = [CAR.model_copy(update={"id": i}) for i in range(6)]
    out, _ = render(Image.new("RGB", (800, 600)), boxes, {"car": "#ff0000"}, 800)
    # left padding of the first spot (above the box), covered again by a fallback caption
    assert out.getpixel((100, 98)) == (255, 0, 0)


def test_render_frame_uses_the_job():
    task = fake_task([shape(1, 1, 0, [0, 0, 5, 5])], get_frame=_frame_png)
    job = gt_job([shape(2, 2, 0, [10, 10, 50, 50])], included=(0, 2))
    client = job_client(task, job)
    _, text = render_frame(client, 1, 0, 1280, job_id=30)
    assert "#2" in text and "#1 " not in text
