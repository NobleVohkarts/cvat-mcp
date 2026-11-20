from PIL import Image

from cvat_mcp.render import clamp_side, legend, place_caption, render
from cvat_mcp.services.annotations import FrameObject

CAR = FrameObject(
    id=1,
    label="car",
    type="rectangle",
    bbox=[100, 100, 300, 250],
    attributes={},
    source="manual",
    occluded=False,
)


def _black():
    return Image.new("RGB", (800, 600))


def test_downscales_to_max_side():
    out, scale = render(_black(), [CAR], {"car": "#ff0000"}, 400)
    assert out.size == (400, 300)
    assert scale == 0.5


def test_box_drawn_at_scaled_position():
    out, _ = render(_black(), [CAR], {"car": "#ff0000"}, 400)
    r, g, _b = out.getpixel((50, 100))  # left edge, mid-height of the scaled box
    assert r > 200 and g < 80
    assert out.getpixel((100, 87)) == (0, 0, 0)  # inside the box stays untouched


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


def test_caption_goes_above_a_free_box():
    assert place_caption((100, 100, 200, 200), (50, 14), (640, 480), []) == (100, 86)


def test_caption_goes_inside_when_the_box_touches_the_top():
    assert place_caption((100, 5, 200, 200), (50, 14), (640, 480), []) == (100, 5)


def test_caption_avoids_taken_places():
    taken = [(100, 86, 150, 100), (100, 100, 150, 114)]
    assert place_caption((100, 100, 200, 200), (50, 14), (640, 480), taken) == (100, 200)


def test_caption_stays_inside_at_the_right_edge():
    x, y = place_caption((620, 100, 640, 200), (50, 14), (640, 480), [])
    assert x + 50 <= 640 and y >= 0


def test_neighbouring_captions_do_not_overlap():
    taken = []
    for box in [(100, 100, 200, 200), (130, 100, 230, 200), (160, 100, 260, 200)]:
        x, y = place_caption(box, (50, 14), (640, 480), taken)
        rect = (x, y, x + 50, y + 14)
        assert all(
            rect[2] <= t[0] or rect[0] >= t[2] or rect[3] <= t[1] or rect[1] >= t[3] for t in taken
        )
        taken.append(rect)


def test_caption_bar_lets_the_image_show_through():
    white = Image.new("RGB", (800, 600), "white")
    out, _ = render(white, [CAR], {"car": "#ff0000"}, 400)
    r, g, b = out.getpixel((50, 47))  # left padding of the caption bar above the scaled box
    assert r > 200 and 60 < g < 230 and 60 < b < 230  # red blended with white, not solid
