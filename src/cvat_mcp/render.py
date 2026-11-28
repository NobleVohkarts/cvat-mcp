"""Draw annotations over a frame so the model can check them by looking."""

import io

from cvat_sdk import Client
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageStat

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.annotations import FrameObject, frame_annotations, scope_of
from cvat_mcp.services.issues import check_bbox
from cvat_mcp.services.tasks import clamp_limit

FALLBACK_COLOR = "#ffffff"


def clamp_side(max_side: int) -> int:
    return clamp_limit(max_side, low=256, high=2048)


MAX_ZOOM = 4.0  # a region is enlarged at most this much, so small crops don't turn to mush

CAPTION_ALPHA = 140  # ~0.55: the frame stays visible under the caption bars

Rect = tuple[float, float, float, float]


def _overlaps(a: Rect, b: Rect) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _touches(box: list[float], region: Rect) -> bool:
    """Inclusive, so points and zero-area boxes inside or on the edge count."""
    return (
        box[0] <= region[2] and region[0] <= box[2] and box[1] <= region[3] and region[1] <= box[3]
    )


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


def resize(image: Image.Image, max_side: int, max_zoom: float = 1.0) -> tuple[Image.Image, float]:
    """Fit into max_side; enlarge at most max_zoom times (1: only ever shrink)."""
    out = image.convert("RGB")
    zoom = min(max_side / max(image.size), max_zoom)
    if zoom > 1:
        out = out.resize((round(image.width * zoom), round(image.height * zoom)), Image.LANCZOS)
    else:
        out.thumbnail((max_side, max_side), Image.LANCZOS)
    return out, out.width / image.width


def clip_region(region: list[float], width: int, height: int) -> Rect:
    """The region [x1, y1, x2, y2] (original pixels) clipped to the frame."""
    check_bbox(region)
    x1, y1, x2, y2 = (
        max(region[0], 0),
        max(region[1], 0),
        min(region[2], width),
        min(region[3], height),
    )
    if x1 >= x2 or y1 >= y2:
        raise InvalidArgument(f"region {region} is outside the frame ({width}x{height} pixels)")
    return x1, y1, x2, y2


def annotate(
    image: Image.Image,
    objects: list[FrameObject],
    colors: dict[str, str],
    scale: float,
    offset: tuple[float, float] = (0.0, 0.0),
) -> Image.Image:
    """A copy of the (already resized) frame with shapes and semi-transparent captions;
    original pixel (x, y) is drawn at ((x - offset x) * scale, (y - offset y) * scale)."""
    out = image.copy()
    side = max(out.size)
    width = max(2, round(side / 400))
    font = ImageFont.load_default(size=max(12, side // 60))
    draw = ImageDraw.Draw(out, "RGBA")  # RGBA fills blend into the RGB frame

    def to_image(values: list[float]) -> list[float]:  # x, y, x, y, ...
        return [(v - offset[i % 2]) * scale for i, v in enumerate(values)]

    boxes = [
        (obj, tuple(to_image(obj.bbox)), colors.get(obj.label, FALLBACK_COLOR)) for obj in objects
    ]
    for obj, box, color in boxes:
        xy = to_image(obj.points or [])
        if not xy:  # rectangles, and shapes without vertices (masks, skeletons, ellipses)
            draw.rectangle(box, outline=color, width=width)
        elif obj.type == "polygon":
            draw.polygon(xy, outline=color, width=width)
        elif obj.type == "polyline":
            draw.line(xy, fill=color, width=width, joint="curve")
        else:  # points
            r = 2 * width
            for x, y in zip(xy[0::2], xy[1::2]):
                draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
    taken: list[Rect] = []
    for obj, box, color in boxes:  # captions after all shapes, so no outline crosses a caption
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


def legend(
    objects: list[FrameObject],
    scale: float,
    frame: int,
    skipped_tracks: int = 0,
    region: Rect | None = None,
) -> str:
    what = f"{len(objects)} objects"
    if region is not None:
        what = f"region [{', '.join(f'{v:g}' for v in region)}], {what} touch it"
    head = (
        f"frame {frame}, {what}, image scaled by {scale:.3f} "
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
    region: list[float] | None = None,
) -> tuple[list[bytes], str]:
    """JPEGs of the frame (or of a region of it, enlarged up to MAX_ZOOM) with its annotations
    drawn (and, optionally, clean), and a legend."""
    task = client.tasks.retrieve(task_id)
    labels = task.get_labels()
    data = frame_annotations(scope_of(client, task, job_id), labels, frame)
    colors = {label.name: label.color for label in labels if label.color}
    objects, rect = data.objects, (0, 0, data.width, data.height)
    if region is not None:  # checked before the frame is downloaded
        rect = clip_region(region, data.width, data.height)
        objects = [o for o in objects if _touches(o.bbox, rect)]
    # "compressed" frames are EXIF-rotated like CVAT's coordinates, and may be smaller than the
    # originals, whose pixels the coordinates are in.
    with Image.open(task.get_frame(frame, quality="compressed")) as image:
        k = image.width / data.width  # original pixels -> this image's pixels
        if region is not None:
            image = image.crop(tuple(round(v * k) for v in rect))
        resized, zoom = resize(image, clamp_side(max_side), MAX_ZOOM if region else 1.0)
    scale = zoom * k  # original pixels -> output pixels
    images = [annotate(resized, objects, colors, scale, rect[:2])]
    text = legend(objects, scale, frame, data.skipped_tracks, rect if region else None)
    if include_clean:
        images.append(resized)
        text = "The second image is the same frame without annotations.\n" + text
    return [_jpeg(image) for image in images], text
