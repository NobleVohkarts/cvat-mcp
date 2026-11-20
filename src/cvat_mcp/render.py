"""Draw annotations over a frame so the model can check them by looking."""

import io

from cvat_sdk import Client
from PIL import Image, ImageColor, ImageDraw, ImageFont

from cvat_mcp.services.annotations import FrameObject, frame_annotations
from cvat_mcp.services.tasks import clamp_limit

FALLBACK_COLOR = "#ffffff"


def clamp_side(max_side: int) -> int:
    return clamp_limit(max_side, low=256, high=2048)


CAPTION_ALPHA = 140  # ~0.55: the frame stays visible under the caption bars

Rect = tuple[float, float, float, float]


def place_caption(
    box: Rect, size: tuple[float, float], image: tuple[int, int], taken: list[Rect]
) -> tuple[float, float]:
    """Top-left of a caption: above the box, inside its top, below it, or inside its bottom.

    The first candidate that fits in the image and doesn't overlap an earlier caption wins;
    otherwise the first that fits, so crowded frames still get every caption.
    """
    (x1, y1, _, y2), (w, h), (width, height) = box, size, image
    x = max(0.0, min(x1, width - w))
    fitting = [(x, y) for y in (y1 - h, y1, y2, y2 - h) if 0 <= y and y + h <= height] or [
        (x, max(0.0, min(y1, height - h)))
    ]

    def free(pos):
        return all(
            pos[0] + w <= t[0] or pos[0] >= t[2] or pos[1] + h <= t[1] or pos[1] >= t[3]
            for t in taken
        )

    return next((pos for pos in fitting if free(pos)), fitting[0])


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
    bars = Image.new("RGBA", out.size, (0, 0, 0, 0))
    bars_draw = ImageDraw.Draw(bars)
    taken: list[Rect] = []
    texts = []
    for obj in objects:
        color = colors.get(obj.label, FALLBACK_COLOR)
        box = tuple(v * scale for v in obj.bbox)
        draw.rectangle(box, outline=color, width=width)
        caption = f"#{obj.id} {obj.label}"
        left, top, right, bottom = draw.textbbox((0, 0), caption, font=font)
        text_w, text_h = right - left + 2 * width, bottom - top + 2 * width
        tx, ty = place_caption(box, (text_w, text_h), out.size, taken)
        taken.append((tx, ty, tx + text_w, ty + text_h))
        bars_draw.rectangle(taken[-1], fill=(*ImageColor.getrgb(color)[:3], CAPTION_ALPHA))
        ink = "black" if ImageColor.getcolor(color, "L") > 140 else "white"
        texts.append(((tx + width - left, ty + width - top), caption, ink))
    out = Image.alpha_composite(out.convert("RGBA"), bars).convert("RGB")
    draw = ImageDraw.Draw(out)
    for position, caption, ink in texts:  # text stays opaque on top of the bars
        draw.text(position, caption, fill=ink, font=font)
    return out, scale


def legend(objects: list[FrameObject], scale: float, frame: int, skipped_tracks: int = 0) -> str:
    head = (
        f"frame {frame}, {len(objects)} objects, image scaled by {scale:.3f} "
        "(coordinates below are original pixels)"
    )
    if skipped_tracks:
        head += f"\n{skipped_tracks} tracked objects are not drawn (tracks are not supported yet)"
    if not objects and not skipped_tracks:
        return f"{head}\nno objects"
    return "\n".join([head, *(f"#{o.id} {o.label} {o.bbox}" for o in objects)])


def _jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def render_frame(
    client: Client, task_id: int, frame: int, max_side: int, include_clean: bool = False
) -> tuple[bytes, bytes | None, str]:
    """The frame with its annotations drawn (JPEG), optionally the same frame clean, and a legend."""
    task = client.tasks.retrieve(task_id)
    labels = task.get_labels()
    data = frame_annotations(task, labels, frame)
    colors = {label.name: label.color for label in labels if label.color}
    # "compressed" frames are EXIF-rotated like CVAT's coordinates, and smaller than originals.
    with Image.open(task.get_frame(frame, quality="compressed")) as image:
        side = clamp_side(max_side)
        out, scale = render(image, data.objects, colors, side)
        clean = render(image, [], colors, side)[0] if include_clean else None
    text = legend(data.objects, scale, frame, data.skipped_tracks)
    if clean is not None:
        text = "The second image is the same frame without annotations.\n" + text
    return _jpeg(out), clean and _jpeg(clean), text
