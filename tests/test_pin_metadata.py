"""Tests for pin metadata (offline, no network)."""

from __future__ import annotations

from typing import Any

from zhihu_cli.content.handlers import people, pin


def _page_pin(**overrides: Any) -> dict[str, Any]:
    """Build a page-state pin entity as ``scrape_pin`` hands it to the parser."""
    item: dict[str, Any] = {
        "id": "123",
        "content": [{"type": "text", "title": "", "content": "标题 | <p>正文</p><br>更多"}],
        "reactionCount": 5,
        "commentCount": 1,
        "created": 1_700_000_000,
        "updated": 1_700_000_100,
        "author": "alice",
    }
    item.update(overrides)
    return item


def test_page_state_pin_excerpt_is_plain_text() -> None:
    """A page-state pin has no ``excerpt``; the block list is flattened to text.

    Regression: the fallback used to slice ``content`` directly, which is a
    list of blocks, so callers printed the raw block list instead of a title.
    """
    meta = pin.parse_pin_metadata(_page_pin())

    assert isinstance(meta["excerpt"], str)
    assert meta["excerpt"] == "标题 | 正文更多"


def test_page_state_pin_excerpt_skips_image_blocks() -> None:
    """Image blocks contribute no text to the excerpt."""
    item = _page_pin(content=[{"type": "image", "url": "https://pic.example/x.jpg"}])

    assert pin.parse_pin_metadata(item)["excerpt"] == ""


def test_string_content_falls_back_to_text() -> None:
    """API-shaped pins may carry a plain string body instead of blocks."""
    meta = pin.parse_pin_metadata({"id": "1", "content": "<p>hello</p> world"})

    assert meta["excerpt"] == "hello world"


def test_existing_excerpt_is_kept() -> None:
    """An explicit ``excerpt`` wins over the body."""
    meta = pin.parse_pin_metadata({"id": "1", "excerpt": "<em>hi</em>", "content": []})

    assert meta["excerpt"] == "<em>hi</em>"


def test_member_pin_null_excerpt_becomes_empty_string() -> None:
    """The member-pin API leaves ``excerpt`` null; the listing returns ``""``."""
    parsed = people._parse_member_pin(
        {"id": "1", "excerpt": None, "content": [{"type": "text", "content": "<p>hi</p>"}]}
    )

    assert parsed["excerpt"] == ""
    assert parsed["content_text"] == "<p>hi</p>"
