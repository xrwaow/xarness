"""Image attachments for user messages.

Loads an image file, downsamples it to at most :data:`MAX_PIXELS` pixels
(Pillow's ``thumbnail`` with LANCZOS resampling, only when it actually
exceeds the cap), and base64-encodes it for the OpenAI-compatible
``image_url`` content part (``data:`` URL). The resulting
:class:`ImageAttachment` is a plain dataclass so it round-trips through
session JSON unchanged — reloading a session never re-processes pixels.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urlparse

from PIL import Image

# Hard pixel budget per image; anything larger is downsampled once, on load.
MAX_PIXELS = 1_000_000

# JPEG quality when (re-)encoding a downsampled JPEG; other formats stay PNG.
_JPEG_QUALITY = 85

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

# The placeholder an attached image leaves in the input text (and the
# transcript). One token per image, deleted atomically in the editor.
_TOKEN_TEXT = "[🖼 {name} {width}×{height}]"
_IMAGE_TOKEN_RE = re.compile(r"\[🖼 [^\]\n]+\]")


class ImageError(Exception):
    """The file could not be read as an image."""


@dataclass(slots=True)
class ImageAttachment:
    """One image attached to a user message.

    ``data_b64`` is the encoded image bytes; ``width``/``height`` describe
    the encoded image (post-downsample) so the TUI can show real dims
    without decoding. Persisted verbatim in session files."""

    name: str
    mime: str
    data_b64: str
    width: int
    height: int

    def data_url(self) -> str:
        return f"data:{self.mime};base64,{self.data_b64}"


def token_text(image: ImageAttachment) -> str:
    """The placeholder token an attached image appears as in the text."""
    return _TOKEN_TEXT.format(
        name=image.name, width=image.width, height=image.height
    )


def find_image_tokens(text: str) -> list[str]:
    """Image placeholder tokens appearing in ``text``, in order."""
    return _IMAGE_TOKEN_RE.findall(text)


def is_image_path(path: Path | str) -> bool:
    """True when the path's suffix marks it as a supported image format."""
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def parse_image_path(text: str) -> Path | None:
    """Best-effort extract a filesystem path from pasted text.

    Terminals paste clipboard *images* as ``file://`` URIs (kitty) or plain
    paths (iTerm2); some wrap the path in quotes. Returns ``None`` when the
    text cannot be a file path (multiline, no scheme-looking content that
    resolves). Percent escapes in URIs are decoded (``%20`` → space).
    """
    text = text.strip().strip("'\"")
    if not text or "\n" in text:
        return None
    if text.startswith("file://"):
        parsed = urlparse(text)
        if parsed.netloc not in ("", "localhost"):
            return None  # a remote file:// reference is not readable here
        return Path(unquote(parsed.path))
    if text.startswith(("http://", "https://")):
        return None
    return Path(text).expanduser()


def load_image(path: Path | str) -> ImageAttachment:
    """Read, downsample if needed, and encode one image file.

    Raises :class:`ImageError` when the file is missing, unreadable, or not
    a decodable image. Animated GIFs keep their first frame."""
    file = Path(path).expanduser()
    try:
        img = Image.open(file)
        img.load()
    except (OSError, ValueError) as exc:
        raise ImageError(f"could not read image {file}: {exc}") from None
    name = file.name

    if img.width * img.height > MAX_PIXELS:
        scale = (MAX_PIXELS / (img.width * img.height)) ** 0.5
        img = img.resize(
            (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
            Image.LANCZOS,
        )
        downsampled = True
    else:
        downsampled = False

    fmt = img.format or "PNG"
    if fmt == "GIF":
        # Animated GIFs keep their first frame, re-encoded as PNG/JPEG below.
        downsampled = True

    if fmt not in ("PNG", "JPEG"):
        # WebP/GIF/BMP: re-encode as a widely-accepted format. Transparency
        # is only meaningful in PNG; JPEG needs a plain RGB image.
        fmt = "PNG" if "A" in img.getbands() else "JPEG"

    if fmt == "JPEG" and img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    if not downsampled and fmt in ("PNG", "JPEG"):
        # Untouched within the pixel budget: send the original bytes so no
        # generation loss sneaks in.
        data = file.read_bytes()
        mime = "image/jpeg" if fmt == "JPEG" else "image/png"
        width, height = img.width, img.height
    else:
        buf = BytesIO()
        try:
            if fmt == "JPEG":
                img.save(buf, "JPEG", quality=_JPEG_QUALITY)
                mime = "image/jpeg"
            else:
                img.save(buf, "PNG")
                mime = "image/png"
        except OSError as exc:  # e.g. a mode Pillow cannot encode
            raise ImageError(f"could not encode image {name}: {exc}") from None
        data = buf.getvalue()
        width, height = img.width, img.height

    return ImageAttachment(
        name=name,
        mime=mime,
        data_b64=base64.b64encode(data).decode("ascii"),
        width=width,
        height=height,
    )
