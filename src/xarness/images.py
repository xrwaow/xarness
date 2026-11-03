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
import subprocess
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urlparse

from PIL import Image
from pydantic import BaseModel

# Hard pixel budget per image; anything larger is downsampled once, on load.
MAX_PIXELS = 1_000_000

# JPEG quality when (re-)encoding a downsampled JPEG; other formats stay PNG.
_JPEG_QUALITY = 85

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

# The placeholder an attached image leaves in the input text (and the
# transcript). One token per image, deleted atomically in the editor. The
# ``#id`` suffix (content-hash prefix) keeps two differently-named/sized
# images from colliding: same name and dimensions does not mean the same
# token.
_TOKEN_TEXT = "[🖼 {name} {width}×{height} #{id}]"
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


class ImageAttachment(ImageRef):
    """One image travelling from a client to the host.

    An :class:`ImageRef` plus the encoded bytes: the reference (including
    its content-hash ``id``) is computed once, on load/attach, not per
    keystroke. The TUI puts these in an upload; the host stores the bytes
    in its blob store and keeps only the ref."""

    data_b64: str

    def data(self) -> bytes:
        return base64.b64decode(self.data_b64)


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


def _token_id(image: ImageRef) -> str:
    """The short content id a token carries (distinguishes same-named images)."""
    return image.id[:4]


def token_text(image: ImageRef) -> str:
    """The placeholder token an attached image appears as in the text."""
    return _TOKEN_TEXT.format(
        name=image.name, width=image.width, height=image.height,
        id=_token_id(image),
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
        data = file.read_bytes()
    except OSError as exc:
        raise ImageError(f"could not read image {file}: {exc}") from None
    return load_image_bytes(data, name=file.name)


def load_image_bytes(data: bytes, name: str = "clipboard.png") -> ImageAttachment:
    """Decode, downsample if needed, and encode raw image bytes.

    Raises :class:`ImageError` when the bytes are not a decodable image.
    Animated GIFs keep their first frame."""
    try:
        img = Image.open(BytesIO(data))
        img.load()
    except (OSError, ValueError) as exc:
        raise ImageError(f"could not read image {name}: {exc}") from None
    return _attachment_from_pil(img, name, data)


def _attachment_from_pil(
    img: Image.Image, name: str, original: bytes,
) -> ImageAttachment:
    """Turn a loaded Pillow image into an :class:`ImageAttachment`.

    ``original`` is the source bytes, reused verbatim when the image was
    untouched and already PNG/JPEG (no generation loss)."""
    fmt = img.format or "PNG"
    resized = img.width * img.height > MAX_PIXELS
    if resized:
        scale = (MAX_PIXELS / (img.width * img.height)) ** 0.5
        img = img.resize(
            (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
            Image.LANCZOS,
        )
    # Keep the original bytes when nothing changed and the format is one
    # every provider accepts. GIF (an animated first frame) and the other
    # formats are re-encoded as PNG (the only one that keeps transparency)
    # or JPEG.
    keep = not resized and fmt in ("PNG", "JPEG")
    if not keep:
        if fmt == "GIF" or fmt not in ("PNG", "JPEG"):
            transparent = "A" in img.getbands() or (
                fmt != "JPEG" and "transparency" in img.info
            )
            fmt = "PNG" if transparent else "JPEG"
        if fmt == "JPEG" and img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = BytesIO()
        try:
            if fmt == "JPEG":
                img.save(buf, "JPEG", quality=_JPEG_QUALITY)
            else:
                img.save(buf, "PNG")
        except OSError as exc:  # e.g. a mode Pillow cannot encode
            raise ImageError(f"could not encode image {name}: {exc}") from None
        data = buf.getvalue()
    else:
        data = original
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return ImageAttachment(
        id=hashlib.sha256(data).hexdigest(),
        name=name,
        mime=mime,
        data_b64=base64.b64encode(data).decode("ascii"),
        width=img.width,
        height=img.height,
    )


def _clip_run(cmd: list[str]) -> bytes:
    """Stdout of a clipboard read command; empty on any failure."""
    try:
        return subprocess.run(cmd, capture_output=True, timeout=3, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return b""


def clipboard_image() -> ImageAttachment | None:
    """Read an image straight from the system clipboard, if one is there.

    Tries the platform tools in turn (wl-paste/xclip on Linux, osascript on
    macOS); ``None`` when no image comes back. Nothing is written to disk."""
    for cmd in (
        ["wl-paste", "--type", "image/png"],
        ["wl-paste"],  # default type (may be text/html for browser copies)
        ["xclip", "-selection", "clipboard", "-o"],
        ["osascript", "-e", "the clipboard as «class PNGf»"],
    ):
        data = _clip_run(cmd)
        if cmd[0] == "osascript":
            # osascript reports the PNG as '«data PNGf<hex>»'.
            hex_ = data.decode("ascii", "ignore").strip().removeprefix("«data PNGf")
            try:
                data = bytes.fromhex(hex_.strip().strip("«»\""))
            except ValueError:
                data = b""
        if sniff_mime(data) != "application/octet-stream":
            try:
                return load_image_bytes(data)
            except ImageError:
                pass
    return None


def clipboard_text() -> str | None:
    """Read plain text from the system clipboard, if there is any."""
    for cmd in (
        ["wl-paste", "--type", "text/plain"],
        ["xclip", "-selection", "clipboard", "-o"],
        ["pbpaste"],
    ):
        text = _clip_run(cmd).decode("utf-8", "replace").strip("\n\r")
        if text:
            return text
    return None
