"""Tests for image attachment support: file loading/downsampling, the
multipart wire format, session persistence, and the TUI's @-mention /
vision gating."""

import base64
import io
import json
import unittest
from pathlib import Path

from PIL import Image

from xarness.config import ProviderProfile
from xarness.conversation import Message
from xarness.images import MAX_PIXELS, ImageAttachment, ImageError, is_image_path, load_image
from xarness.session_store import _message_from, _message_dump


def _png(width: int, height: int, color=(255, 0, 0)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, "PNG")
    return buf.getvalue()


class TestLoadImage(unittest.TestCase):
    def test_small_png_passes_through_unchanged(self) -> None:
        import tempfile
        from pathlib import Path

        data = _png(10, 10)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(data)
            path = f.name
        try:
            att = load_image(path)
            self.assertEqual(base64.b64decode(att.data_b64), data)
            self.assertEqual((att.width, att.height), (10, 10))
            self.assertEqual(att.mime, "image/png")
            self.assertEqual(att.name, Path(path).name)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_large_image_is_downsampled_to_1mp(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(_png(2000, 1000))  # 2 MP
            path = f.name
        try:
            att = load_image(path)
            self.assertLessEqual(att.width * att.height, MAX_PIXELS)
            self.assertEqual(att.width, 1414)
            self.assertEqual(att.height, 707)
            # Aspect ratio preserved.
            self.assertAlmostEqual(att.width / att.height, 2.0, places=1)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_unreadable_file_raises(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"not an image")
            path = f.name
        try:
            with self.assertRaises(ImageError):
                load_image(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_is_image_path(self) -> None:
        self.assertTrue(is_image_path("shot.PNG"))
        self.assertTrue(is_image_path("a/b.jpg"))
        self.assertFalse(is_image_path("notes.txt"))
        self.assertFalse(is_image_path(""))

    def test_parse_image_path(self) -> None:
        from xarness.images import parse_image_path

        self.assertEqual(parse_image_path("/tmp/a.png"), Path("/tmp/a.png"))
        self.assertEqual(parse_image_path("~/b.jpg"), Path("~/b.jpg").expanduser())
        self.assertEqual(
            parse_image_path("file:///home/xr/Downloads/qp1dnodvoath1.png"),
            Path("/home/xr/Downloads/qp1dnodvoath1.png"),
        )
        self.assertEqual(
            parse_image_path('file:///home/xr/my%20pics/c.png'),
            Path("/home/xr/my pics/c.png"),
        )
        self.assertEqual(parse_image_path('"/tmp/d.png"'), Path("/tmp/d.png"))
        self.assertIsNone(parse_image_path("https://x.test/e.png"))
        self.assertIsNone(parse_image_path("file://otherhost/f.png"))
        self.assertIsNone(parse_image_path("line1\nline2"))


class TestWireFormat(unittest.TestCase):
    def test_user_message_with_images_emits_content_parts(self) -> None:
        att = ImageAttachment(
            name="shot.png", mime="image/png",
            data_b64=base64.b64encode(_png(4, 4)).decode(), width=4, height=4,
        )
        message = Message(role="user", content="look at this", images=[att])
        wire = message.to_wire()
        self.assertIsInstance(wire["content"], list)
        self.assertEqual(wire["content"][0], {"type": "text", "text": "look at this"})
        image_part = wire["content"][1]
        self.assertEqual(image_part["type"], "image_url")
        url = image_part["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/png;base64,"))

    def test_text_only_message_stays_a_string(self) -> None:
        self.assertEqual(Message(role="user", content="hi").to_wire()["content"], "hi")

    def test_empty_content_with_images_still_emits_image_parts(self) -> None:
        att = ImageAttachment(
            name="s.jpg", mime="image/jpeg", data_b64="aGk=", width=2, height=2,
        )
        wire = Message(role="user", content="", images=[att]).to_wire()
        self.assertEqual([p["type"] for p in wire["content"]], ["image_url"])

    def test_profile_supports_vision_defaults_false(self) -> None:
        profile = ProviderProfile(base_url="https://x.test/v1", model_id="m")
        self.assertFalse(profile.supports_vision)


class TestOrderedWireFormat(unittest.TestCase):
    def test_tokens_order_the_content_blocks(self) -> None:
        a = ImageAttachment(name="a.png", mime="image/png", data_b64="QQ==", width=2, height=2)
        b = ImageAttachment(name="b.png", mime="image/png", data_b64="Qg==", width=3, height=3)
        message = Message(
            role="user",
            content=f"compare [🖼 a.png 2×2] with [🖼 b.png 3×3]",
            images=[a, b],
        )
        parts = message.to_wire()["content"]
        self.assertEqual(
            [(p["type"], p.get("text") or p["image_url"]["url"][-4:]) for p in parts],
            [
                ("text", "compare "),
                ("image_url", "QQ=="),
                ("text", " with "),
                ("image_url", "Qg=="),
            ],
        )

    def test_images_without_tokens_are_appended(self) -> None:
        a = ImageAttachment(name="a.png", mime="image/png", data_b64="QQ==", width=2, height=2)
        message = Message(role="user", content="look", images=[a])
        parts = message.to_wire()["content"]
        self.assertEqual(parts[0]["type"], "text")
        self.assertEqual(parts[1]["type"], "image_url")

    def test_token_without_attachment_keeps_its_text(self) -> None:
        a = ImageAttachment(name="a.png", mime="image/png", data_b64="QQ==", width=2, height=2)
        message = Message(role="user", content="[🖼 a.png 2×2] and [🖼 gone.png 1×1]", images=[a])
        parts = message.to_wire()["content"]
        self.assertEqual(parts[0]["type"], "image_url")
        texts = [p.get("text", "") for p in parts if p["type"] == "text"]
        self.assertTrue(any("gone.png" in t for t in texts))


class TestPersistence(unittest.TestCase):
    def test_images_round_trip_through_session_json(self) -> None:
        att = ImageAttachment(
            name="shot.png", mime="image/png",
            data_b64=base64.b64encode(_png(4, 4)).decode(), width=4, height=4,
        )
        message = Message(role="user", content="see attached", images=[att])
        raw = json.loads(json.dumps(_message_dump(message)))
        loaded = _message_from(raw)
        self.assertEqual(loaded.images, [att])
        self.assertEqual(loaded.to_wire()["content"][1]["type"], "image_url")


if __name__ == "__main__":
    unittest.main()