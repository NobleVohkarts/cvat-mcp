"""Draw annotations over a frame so the model can check them by looking."""

import io

from cvat_sdk import Client
from PIL import Image, ImageColor, ImageDraw, ImageFont

from cvat_mcp.services.annotations import FrameObject, get_frame_annotations
from cvat_mcp.services.tasks import clamp_limit

FALLBACK_COLOR = "#ffffff"


def clamp_side(max_side: int) -> int:
    return clamp_limit(max_side, low=256, high=2048)


def scale_box(bbox: list[float], scale: float) -> list[float]:
    return [v * scale for v in bbox]


def render(
    image: Image.Image, objects: list[FrameObject], colors: dict[str, str], max_side: int
) -> tuple[Image.Image, float]:
    """Downscale first, then draw at the output size so lines and captions stay readable."""
    scale = min(1.0, max_side / max(image.size))
    out = image.convert("RGB")
    if scale < 1:
        out = out.resize((round(out.width * scale), round(out.height * scale)), Image.LANCZOS)
    side = max(out.size)
    width = max(2, round(side / 400))
    font = ImageFont.load_default(size=max(12, side // 60))
    draw = ImageDraw.Draw(out)
    for obj in objects:
        color = colors.get(obj.label, FALLBACK_COLOR)
        x1, y1, x2, y2 = scale_box(obj.bbox, scale)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        caption = f"#{obj.id} {obj.label}"
        left, top, right, bottom = draw.textbbox((0, 0), caption, font=font)
        text_w, text_h = right - left + 2 * width, bottom - top + 2 * width
        ty = y1 - text_h if y1 >= text_h else y1  # above the box, or inside at the image top
        draw.rectangle([x1, ty, x1 + text_w, ty + text_h], fill=color)
        r, g, b = ImageColor.getrgb(color)[:3]
        ink = "black" if 0.299 * r + 0.587 * g + 0.114 * b > 140 else "white"
        draw.text((x1 + width - left, ty + width - top), caption, fill=ink, font=font)
    return out, scale


def legend(objects: list[FrameObject], scale: float, frame: int) -> str:
    head = (
        f"frame {frame}, {len(objects)} objects, image scaled by {scale:.3f} "
        "(coordinates below are original pixels)"
    )
    if not objects:
        return f"{head}\nno objects"
    return "\n".join([head, *(f"#{o.id} {o.label} {[float(v) for v in o.bbox]}" for o in objects)])


def render_frame(client: Client, task_id: int, frame: int, max_side: int) -> tuple[bytes, str]:
    """JPEG bytes of the frame with its annotations drawn, and a text legend."""
    data = get_frame_annotations(client, task_id, frame)
    task = client.tasks.retrieve(task_id)
    colors = {label.name: label.color for label in task.get_labels() if label.color}
    with Image.open(task.get_frame(frame, quality="original")) as image:
        out, scale = render(image, data.objects, colors, clamp_side(max_side))
    buffer = io.BytesIO()
    out.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue(), legend(data.objects, scale, frame)
