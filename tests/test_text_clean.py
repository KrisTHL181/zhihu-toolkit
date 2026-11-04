"""Tests for the shared text-cleaning helpers."""

from __future__ import annotations

from zhihu_cli.content.utils.text_clean import (
    body_after_frontmatter,
    parse_frontmatter,
    strip_html_tags,
)


def test_body_after_frontmatter_current_format() -> None:
    """Current YAML frontmatter is stripped."""
    text = "---\ntitle: hello\n---\nbody text"

    assert body_after_frontmatter(text) == "\nbody text"


def test_body_after_frontmatter_legacy_format() -> None:
    """Legacy JSON metadata before the delimiter is stripped."""
    text = '{"title": "hello"}\n---\nbody text'

    assert body_after_frontmatter(text) == "\nbody text"


def test_body_after_frontmatter_no_frontmatter() -> None:
    """Text without a delimiter is returned unchanged."""
    text = "just body text"

    assert body_after_frontmatter(text) is text


def test_body_after_frontmatter_matches_parse_frontmatter() -> None:
    """Body extraction agrees with the metadata parser on both formats."""
    current = "---\ntitle: hello\n---\nbody"
    legacy = '{"title": "hello"}\n---\nbody'

    for text in (current, legacy):
        assert parse_frontmatter(text) == {"title": "hello"}
        assert "body" in body_after_frontmatter(text)


def test_strip_html_tags_default_keeps_entities() -> None:
    """The default drops tags but leaves HTML entities untouched."""
    assert strip_html_tags("<p>a &amp; b</p>") == "a &amp; b"


def test_strip_html_tags_unescape() -> None:
    """``unescape=True`` decodes entities after removing the tags."""
    assert strip_html_tags("<p>a &amp; b</p>", unescape=True) == "a & b"
