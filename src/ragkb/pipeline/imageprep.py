"""Image preprocessing for the vision layer.

The gateway (both Anthropic and Gemini dialects) rejects images whose payload
exceeds a size cap — Claude is explicit: `image exceeds 5 MB maximum`. The real
material includes multi-MB screenshots and 17–18 MB animated GIFs (screen
recordings), which 400 out and leave that image unread.

This module makes any source image fit before it's sent, without losing the
diagnostic content:
- Animated GIF → its first frame (a recording's first frame is a static
  screenshot; we transcribe text/'状态, not motion).
- Oversize raster → progressively downscale + re-encode (PNG, then JPEG) until
  under the byte budget.

Pillow is an optional dependency: if it's missing, images already within budget
pass through untouched, and oversize ones are returned as-is (the caller records
the read failure rather than crashing the run).
"""
from __future__ import annotations

import io
import logging

log = logging.getLogger(__name__)

# Stay comfortably under the gateway's 5 MB cap. The limit applies to the decoded
# image bytes; we target 4.5 MB of raw bytes so there's margin.
_MAX_BYTES = 4_500_000
# Never upscale; only shrink. Cap the longest edge so huge screenshots also lose
# pixel bulk, not just re-encode.
_MAX_EDGE = 2200


def fit_image(data: bytes, media_type: str) -> tuple[bytes, str]:
    """Return (bytes, media_type) within the gateway's limits when possible.

    The gateway rejects an image for EITHER reason: payload too large (>5 MB) OR
    pixel dimensions too large (a long edge beyond the model's cap — an 8192px-wide
    banner is only 0.2 MB yet still 400s). So the decision to preprocess must gate
    on BOTH bytes and dimensions, not bytes alone. Also converts animated GIF to
    its first frame. Falls back to the original bytes if Pillow is unavailable or
    processing fails (caller records the read failure rather than crashing).
    """
    is_gif = media_type == "image/gif"
    small_bytes = len(data) <= _MAX_BYTES
    # Cheap header-only dimension probe (PIL.open is lazy; it doesn't decode pixels
    # until needed). If we can't read the size, fall through to the full path.
    over_pixels = False
    if small_bytes and not is_gif:
        try:
            from PIL import Image as PILImage
            with PILImage.open(io.BytesIO(data)) as probe:
                w, h = probe.size
            over_pixels = max(w, h) > _MAX_EDGE
        except Exception:  # noqa: BLE001 - can't size it → let full path try
            over_pixels = False
        if not over_pixels:
            return data, media_type  # within both byte and pixel budgets
    try:
        from PIL import Image as PILImage
    except ImportError:
        log.warning("Pillow not installed; cannot shrink oversize image "
                    "(%d bytes, %s) — sending as-is", len(data), media_type)
        return data, media_type
    try:
        im = PILImage.open(io.BytesIO(data))
        # GIF/animated → first frame as a static image.
        if getattr(im, "is_animated", False) or is_gif:
            im.seek(0)
        im = im.convert("RGB")
        # Downscale the longest edge first if the image is huge.
        w, h = im.size
        longest = max(w, h)
        if longest > _MAX_EDGE:
            scale = _MAX_EDGE / longest
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        # Encode, shrinking further until within budget. PNG first (lossless for
        # UI/text screenshots), then JPEG at decreasing quality if still too big.
        png = _encode(im, "PNG")
        if len(png) <= _MAX_BYTES:
            return png, "image/png"
        for quality in (85, 70, 55, 40):
            jpg = _encode(im, "JPEG", quality=quality)
            if len(jpg) <= _MAX_BYTES:
                return jpg, "image/jpeg"
            # Still too big → halve dimensions and retry JPEG once more.
        w2, h2 = im.size
        im = im.resize((max(1, w2 // 2), max(1, h2 // 2)))
        jpg = _encode(im, "JPEG", quality=60)
        return (jpg, "image/jpeg") if len(jpg) <= _MAX_BYTES else (jpg, "image/jpeg")
    except Exception as exc:  # noqa: BLE001 - preprocessing must never crash a run
        log.warning("image preprocessing failed (%s); sending original: %s",
                    media_type, exc)
        return data, media_type


def _encode(im, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format=fmt, **kw)
    return buf.getvalue()
