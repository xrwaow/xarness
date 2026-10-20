"""Image attachments for user messages.

Loads an image file, downsamples it to at most :data:`MAX_PIXELS` pixels
(Pillow's ``thumbnail`` with LANCZOS resampling, only when it actually
exceeds the cap), and base64-encodes it for the OpenAI-compatible
``image_url`` content part (``data:`` URL). The resulting
:class:`ImageAttachment` is a pydantic model so it round-trips through
session JSON and the wire protocol unchanged — reloading a session never
re-processes pixels.
"""

from __future__ import annotations

import base64
import hashlib
import re
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urlparse

from PIL import Image
from pydantic import BaseModel, ConfigDict, PrivateAttr

# Hard pixel budget per image; anything larger is downsampled once, on load.
MAX_PIXELS = 1_000_000

# JPEG quality when (re-)encoding a downsampled JPEG; other formats stay PNG.
_JPEG_QUALITY = 85

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

# The placeholder an attached image leaves in the input text (and the
# transcript). One token per image, deleted atomically in the editor. The
# ``#id`` suffix (content-hash prefix) keeps two differently-named/sized
# images from colliding: same name and dimensions no longer means the same
# token. Older sessions carry the suffix-less legacy form, still matched by
# ``IMAGE_TOKEN_RE``.
_TOKEN_TEXT = "[🖼 {name} {width}×{height} #{id}]"
_LEGACY_TOKEN_TEXT = "[🖼 {name} {width}×{height}]"
# Single source of truth for the token shape (see conversation.to_wire).
IMAGE_TOKEN_RE = re.compile(r"\[🖼 [^\]\n]+\]")


class ImageError(Exception):
    """The file could not be read as an image."""


class ImageRef(BaseModel):
    """An image attached to a message, by content hash.

    This is the persisted/wire shape: the bytes live once in the session's
    ``blobs/<id>`` directory and are served by
    ``GET /v1/sessions/{name}/images/{id}``. Persisted JSON and snapshots
    therefore never carry base64 payloads."""

    id: str
    name: str
    mime: str
    width: int
    height: int


class ImageAttachment(BaseModel):
    """One image travelling from a client to the host.

    ``data_b64`` is the encoded image bytes; ``width``/``height`` describe
    the encoded image (post-downsample) so the TUI can show real dims
    without decoding. ``SendMessage`` carries these; the host stores the
    bytes in its blob store and keeps only an :class:`ImageRef`."""

    model_config = ConfigDict(extra="ignore")

    name: str
    mime: str
    data_b64: str
    width: int
    height: int

    # Content-hash id, computed lazily (hashing a multi-MB payload on every
    # keystroke would be wasteful).
    _ref_id: str | None = PrivateAttr(default=None)

    @property
    def ref_id(self) -> str:
        if self._ref_id is None:
            self._ref_id = self.to_ref().id
        return self._ref_id

    def data(self) -> bytes:
        return base64.b64decode(self.data_b64)

    def to_ref(self) -> ImageRef:
        """The content-addressed reference for these bytes."""
        return ImageRef(
            id=hashlib.sha256(self.data()).hexdigest(),
            name=self.name, mime=self.mime,
            width=self.width, height=self.height,
        )


def sniff_mime(data: bytes) -> str:
    """Best-effort MIME type from a blob's magic bytes."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:2] == b"BM":
        return "image/bmp"
    return "application/octet-stream"


def _token_id(image: ImageRef | ImageAttachment) -> str:
    """The short content id a token carries (distinguishes same-named images)."""
    return (image.id if isinstance(image, ImageRef) else image.ref_id)[:4]


def token_text(image: ImageRef | ImageAttachment) -> str:
    """The placeholder token an attached image appears as in the text."""
    return _TOKEN_TEXT.format(
        name=image.name, width=image.width, height=image.height,
        id=_token_id(image),
    )


def legacy_token_text(image: ImageRef | ImageAttachment) -> str:
    """The pre-T1.11 token form (no ``#id``), for reading old sessions."""
    return _LEGACY_TOKEN_TEXT.format(
        name=image.name, width=image.width, height=image.height,
    )


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
    fmt = img.format or "PNG"

    if img.width * img.height > MAX_PIXELS:
        scale = (MAX_PIXELS / (img.width * img.height)) ** 0.5
        img = img.resize(
            (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
            Image.LANCZOS,
        )
        downsampled = True
    else:
        downsampled = False

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
