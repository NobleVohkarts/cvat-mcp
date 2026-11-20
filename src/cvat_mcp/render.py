"""Draw annotations over a frame so the model can check them by looking."""

import io

from cvat_sdk import Client
from PIL import Image, ImageColor, ImageDraw, ImageFont

from cvat_mcp.services.annotations import FrameObject, frame_annotations
from cvat_mcp.services.tasks import clamp_limit

FALLBACK_COLOR = "#ffffff"


def clamp_side(max_side: int) -> int:
    return clamp_limit(max_side, low=256, high=2048)


def render(
    image: Image.Image, objects: list[FrameObject], colors: dict[str, str], max_side: int
) -> tuple[Image.Image, float]:
    """Downscale first, then draw at the output size so lines and captions stay readable."""
    out = image.convert("RGB")
    out.thumbnail((max_side, max_side), Image.LANCZOS)  # only ever shrinks
    scale = out.width / image.width
    side = max(out.size)
    width = max(2, round(side / 400))
    font = ImageFont.load_default(size=max(12, side // 60))
    draw = ImageDraw.Draw(out)
    for obj in objects:
        color = colors.get(obj.label, FALLBACK_COLOR)
        x1, y1, x2, y2 = (v * scale for v in obj.bbox)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        caption = f"#{obj.id} {obj.label}"
        left, top, right, bottom = draw.textbbox((0, 0), caption, font=font)
        text_w, text_h = right - left + 2 * width, bottom - top + 2 * width
        ty = y1 - text_h if y1 >= text_h else y1  # above the box, or inside at the image top
        draw.rectangle([x1, ty, x1 + text_w, ty + text_h], fill=color)
        ink = "black" if ImageColor.getcolor(color, "L") > 140 else "white"
        draw.text((x1 + width - left, ty + width - top), caption, fill=ink, font=font)
    return out, scale


def legend(objects: list[FrameObject], scale: float, frame: int) -> str:
    head = (
        f"frame {frame}, {len(objects)} objects, image scaled by {scale:.3f} "
        "(coordinates below are original pixels)"
    )
    if not objects:
        return f"{head}\nno objects"
    return "\n".join([head, *(f"#{o.id} {o.label} {o.bbox}" for o in objects)])


def render_frame(client: Client, task_id: int, frame: int, max_side: int) -> tuple[bytes, str]:
    """JPEG bytes of the frame with its annotations drawn, and a text legend."""
    task = client.tasks.retrieve(task_id)
    labels = task.get_labels()
    data = frame_annotations(task, labels, frame)
    colors = {label.name: label.color for label in labels if label.color}
    # "compressed" frames are EXIF-rotated like CVAT's coordinates, and smaller than originals.
    with Image.open(task.get_frame(frame, quality="compressed")) as image:
        out, scale = render(image, data.objects, colors, clamp_side(max_side))
    buffer = io.BytesIO()
    out.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue(), legend(data.objects, scale, frame)
