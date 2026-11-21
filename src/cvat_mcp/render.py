"""Draw annotations over a frame so the model can check them by looking."""

import io

from cvat_sdk import Client
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageStat

from cvat_mcp.services.annotations import FrameObject, frame_annotations, scope_of
from cvat_mcp.services.tasks import clamp_limit

FALLBACK_COLOR = "#ffffff"


def clamp_side(max_side: int) -> int:
    return clamp_limit(max_side, low=256, high=2048)


CAPTION_ALPHA = 140  # ~0.55: the frame stays visible under the caption bars

Rect = tuple[float, float, float, float]


def _overlaps(a: Rect, b: Rect) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def place_caption(
    box: Rect, size: tuple[float, float], image: tuple[int, int], taken: list[Rect]
) -> tuple[float, float]:
    """Top-left of a caption: above the box, inside its top, below it, or inside its bottom.

    The first candidate that fits in the image and doesn't overlap an earlier caption wins;
    otherwise the first that fits, so crowded frames still get every caption.
    """
    (x1, y1, _, y2), (w, h), (width, height) = box, size, image
    x = max(0.0, min(x1, width - w))
    ys = [y for y in (y1 - h, y1, y2, y2 - h) if 0 <= y <= height - h]
    fitting = [(x, y) for y in ys] or [(x, max(0.0, min(y1, height - h)))]
    for tx, ty in fitting:
        if not any(_overlaps((tx, ty, tx + w, ty + h), t) for t in taken):
            return tx, ty
    return fitting[0]


def resize(image: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    out = image.convert("RGB")
    out.thumbnail((max_side, max_side), Image.LANCZOS)  # only ever shrinks
    return out, out.width / image.width


def annotate(
    image: Image.Image, objects: list[FrameObject], colors: dict[str, str], scale: float
) -> Image.Image:
    """A copy of the (already resized) frame with boxes and semi-transparent captions."""
    out = image.copy()
    side = max(out.size)
    width = max(2, round(side / 400))
    font = ImageFont.load_default(size=max(12, side // 60))
    draw = ImageDraw.Draw(out, "RGBA")  # RGBA fills blend into the RGB frame
    boxes = [(obj, tuple(v * scale for v in obj.bbox)) for obj in objects]
    for obj, box in boxes:
        draw.rectangle(box, outline=colors.get(obj.label, FALLBACK_COLOR), width=width)
    taken: list[Rect] = []
    for obj, box in boxes:  # captions after all boxes, so no outline crosses a caption
        color = colors.get(obj.label, FALLBACK_COLOR)
        caption = f"#{obj.id} {obj.label}"
        left, top, right, bottom = draw.textbbox((0, 0), caption, font=font)
        text_w, text_h = right - left + 2 * width, bottom - top + 2 * width
        tx, ty = place_caption(box, (text_w, text_h), out.size, taken)
        rect = (tx, ty, tx + text_w, ty + text_h)
        # A caption that has to overlap an earlier one is opaque, so the two texts don't mix.
        alpha = 255 if any(_overlaps(rect, t) for t in taken) else CAPTION_ALPHA
        taken.append(rect)
        rgb = ImageColor.getrgb(color)[:3]
        under = ImageStat.Stat(out.crop(tuple(round(v) for v in rect))).mean[:3]
        bar = [(c * alpha + u * (255 - alpha)) / 255 for c, u in zip(rgb, under)]
        draw.rectangle(rect, fill=(*rgb, alpha))
        # pick the ink against the colour the bar really has after blending
        ink = "black" if 0.299 * bar[0] + 0.587 * bar[1] + 0.114 * bar[2] > 140 else "white"
        draw.text((tx + width - left, ty + width - top), caption, fill=ink, font=font)
    return out


def render(
    image: Image.Image, objects: list[FrameObject], colors: dict[str, str], max_side: int
) -> tuple[Image.Image, float]:
    """Downscale first, then draw at the output size so lines and captions stay readable."""
    resized, scale = resize(image, max_side)
    return annotate(resized, objects, colors, scale), scale


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
    client: Client,
    task_id: int,
    frame: int,
    max_side: int,
    include_clean: bool = False,
    job_id: int | None = None,
) -> tuple[list[bytes], str]:
    """JPEGs of the frame with its annotations drawn (and, optionally, clean), and a legend."""
    task = client.tasks.retrieve(task_id)
    labels = task.get_labels()
    data = frame_annotations(scope_of(client, task, job_id), labels, frame)
    colors = {label.name: label.color for label in labels if label.color}
    # "compressed" frames are EXIF-rotated like CVAT's coordinates, and smaller than originals.
    with Image.open(task.get_frame(frame, quality="compressed")) as image:
        resized, scale = resize(image, clamp_side(max_side))
    images = [annotate(resized, data.objects, colors, scale)]
    text = legend(data.objects, scale, frame, data.skipped_tracks)
    if include_clean:
        images.append(resized)
        text = "The second image is the same frame without annotations.\n" + text
    return [_jpeg(image) for image in images], text
