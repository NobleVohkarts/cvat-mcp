from PIL import Image

from cvat_mcp.render import clamp_side, legend, render
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


def test_clamp_side():
    assert clamp_side(10) == 256 and clamp_side(5000) == 2048 and clamp_side(1280) == 1280
