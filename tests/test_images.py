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
from xarness.conversation import Message, build_parts
from xarness.images import (
    MAX_PIXELS, ImageAttachment, ImageError, is_image_path, load_image, token_text,
)
from xarness import session_store
from xarness.session_store import (
    SessionImageStore, _message_from, _message_dump, load_blob,
)


def _user_message(text: str, images: list[ImageAttachment]) -> Message:
    return Message(role="user", content=text, parts=build_parts(text, images))


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
        message = _user_message("look at this", [att])
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
        wire = _user_message("", [att]).to_wire()
        self.assertEqual([p["type"] for p in wire["content"]], ["image_url"])

    def test_profile_supports_vision_defaults_false(self) -> None:
        profile = ProviderProfile(base_url="https://x.test/v1", model_id="m")
        self.assertFalse(profile.supports_vision)


class TestOrderedWireFormat(unittest.TestCase):
    def test_tokens_order_the_content_blocks(self) -> None:
        a = ImageAttachment(name="a.png", mime="image/png", data_b64="QQ==", width=2, height=2)
        b = ImageAttachment(name="b.png", mime="image/png", data_b64="Qg==", width=3, height=3)
        message = _user_message(
            "compare [🖼 a.png 2×2] with [🖼 b.png 3×3]", [a, b],
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
        message = _user_message("look", [a])
        parts = message.to_wire()["content"]
        self.assertEqual(parts[0]["type"], "text")
        self.assertEqual(parts[1]["type"], "image_url")

    def test_token_without_attachment_keeps_its_text(self) -> None:
        a = ImageAttachment(name="a.png", mime="image/png", data_b64="QQ==", width=2, height=2)
        message = _user_message("[🖼 a.png 2×2] and [🖼 gone.png 1×1]", [a])
        parts = message.to_wire()["content"]
        self.assertEqual(parts[0]["type"], "image_url")
        texts = [p.get("text", "") for p in parts if p["type"] == "text"]
        self.assertTrue(any("gone.png" in t for t in texts))

    def test_same_name_and_size_tokens_do_not_collide(self) -> None:
        """Two different images with equal name/dimensions get distinct
        tokens, so both reach the wire instead of one turning into text."""
        a = ImageAttachment(name="x.png", mime="image/png", data_b64="QQ==", width=2, height=2)
        b = ImageAttachment(name="x.png", mime="image/png", data_b64="Qg==", width=2, height=2)
        self.assertNotEqual(token_text(a), token_text(b))
        text = f"{token_text(a)} vs {token_text(b)}"
        parts = _user_message(text, [a, b]).to_wire()["content"]
        kinds = [p["type"] for p in parts]
        self.assertEqual(kinds.count("image_url"), 2, kinds)


class TestPersistence(unittest.TestCase):
    def test_images_round_trip_through_session_json(self) -> None:
        att = ImageAttachment(
            name="shot.png", mime="image/png",
            data_b64=base64.b64encode(_png(4, 4)).decode(), width=4, height=4,
        )
        message = _user_message("see attached", [att])
        raw = json.loads(json.dumps(_message_dump(message)))
        loaded = _message_from(raw)
        # The persisted shape carries only the content hash, never base64.
        self.assertNotIn("data_b64", json.dumps(raw))
        self.assertNotIn("data", json.dumps(raw))
        ref = loaded.parts[1].ref
        self.assertEqual(ref.id, att.to_ref().id)
        # Bytes are resolved from the store when turning into a wire message.
        wire = loaded.to_wire(image_loader=lambda _id: att.data())
        self.assertEqual(wire["content"][1]["type"], "image_url")

    def test_snapshot_carries_refs_not_bytes(self) -> None:
        from xarness.protocol import Snapshot

        att = ImageAttachment(
            name="shot.png", mime="image/png", data_b64="aGk=", width=2, height=2,
        )
        message = _user_message("see attached", [att])
        payload = Snapshot(
            messages=[message], status={"state": "idle"},
        ).model_dump_json()
        self.assertNotIn(att.data_b64, payload)
        self.assertIn(att.to_ref().id, payload)

    def test_legacy_inline_images_migrate_to_refs(self) -> None:
        att = ImageAttachment(
            name="shot.png", mime="image/png",
            data_b64=base64.b64encode(_png(4, 4)).decode(), width=4, height=4,
        )
        raw = {
            "role": "user", "content": "see attached",
            "images": [att.model_dump()],
        }
        loaded = _message_from(raw)
        self.assertFalse(hasattr(loaded, "images"))
        self.assertEqual(loaded.parts[1].ref.id, att.to_ref().id)


class TestBlobStore(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())
        self._old = session_store.SESSIONS_DIR
        session_store.SESSIONS_DIR = self.tmp / "sessions"

    def tearDown(self) -> None:
        session_store.SESSIONS_DIR = self._old
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _attachment(self, data: bytes = b"cat") -> ImageAttachment:
        return ImageAttachment(
            name="cat.png", mime="image/png",
            data_b64=base64.b64encode(data).decode(), width=2, height=2,
        )

    def test_save_is_content_addressed_and_deduped(self) -> None:
        store = SessionImageStore("s1")
        first = store.save(self._attachment())
        second = store.save(self._attachment())
        self.assertEqual(first.id, second.id)
        blobs = list((self.tmp / "sessions" / "s1" / "blobs").iterdir())
        self.assertEqual(len(blobs), 1)
        self.assertEqual(store.load(first.id), b"cat")
        self.assertEqual(load_blob("s1", first.id), b"cat")
        self.assertIsNone(load_blob("s1", "../escape"))

    def test_session_file_has_no_base64_payloads(self) -> None:
        conversation = session_store.Conversation()
        att = self._attachment()
        conversation.add(Message(
            role="user", content="look [x]",
            parts=build_parts("look [x]", [att], SessionImageStore("s1")),
        ))
        state = session_store.SessionState(conversation=conversation)
        session_store.save_state("s1", state)
        raw = (self.tmp / "sessions" / "s1.json").read_text()
        self.assertNotIn(att.data_b64, raw)
        self.assertNotIn("data_b64", raw)
        loaded = session_store.load_state("s1")
        ref = loaded.conversation.messages[0].parts[1].ref
        self.assertEqual(ref.id, att.to_ref().id)
        # The provider wire still gets the bytes, resolved from the blob.
        wire = loaded.conversation.to_wire(
            image_loader=lambda image_id: load_blob("s1", image_id),
        )
        self.assertEqual(wire[0]["content"][1]["type"], "image_url")

    def test_api_serves_the_blob(self) -> None:
        import asyncio

        import httpx

        from xarness.api import create_app

        data = _png(2, 2)
        ref = SessionImageStore("s1").save(ImageAttachment(
            name="cat.png", mime="image/png",
            data_b64=base64.b64encode(data).decode(), width=2, height=2,
        ))
        app = create_app(None)  # the image route needs only the blob store

        async def fetch(image_id: str) -> httpx.Response:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test",
            ) as client:
                return await client.get(
                    f"/v1/sessions/s1/images/{image_id}",
                )

        response = asyncio.run(fetch(ref.id))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, data)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertEqual(asyncio.run(fetch("0" * 64)).status_code, 404)

    def test_no_session_store_keeps_bytes_in_memory(self) -> None:
        store = SessionImageStore("s2", persist=False)
        ref = store.save(self._attachment())
        self.assertEqual(store.load(ref.id), b"cat")
        self.assertFalse((self.tmp / "sessions" / "s2").exists())


if __name__ == "__main__":
    unittest.main()