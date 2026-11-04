"""Tests for the shared Zhihu URL helpers."""

from __future__ import annotations

from zhihu_cli.content.handlers import (
    canonical_answer_url,
    canonical_pin_url,
    split_composite_id,
)


def test_canonical_answer_url_short_link() -> None:
    """A short answer URL is normalised."""
    assert canonical_answer_url("https://www.zhihu.com/answer/123") == "https://www.zhihu.com/answer/123"


def test_canonical_answer_url_bare_id() -> None:
    """A bare numeric ID is expanded."""
    assert canonical_answer_url("123") == "https://www.zhihu.com/answer/123"


def test_canonical_answer_url_composite_with_query() -> None:
    """Question/answer paths ignore query strings."""
    assert canonical_answer_url("https://www.zhihu.com/question/1/answer/2?utm=x") == ("https://www.zhihu.com/answer/2")


def test_canonical_answer_url_unmatched_is_stripped() -> None:
    """Unparseable input is returned stripped, not raised."""
    assert canonical_answer_url("  https://example.com/foo  ") == "https://example.com/foo"


def test_canonical_pin_url_bare_id() -> None:
    """A bare pin ID becomes a full URL."""
    assert canonical_pin_url("abc") == "https://www.zhihu.com/pin/abc"


def test_canonical_pin_url_full_url_with_query() -> None:
    """Pin URLs are normalised and query strings dropped."""
    assert canonical_pin_url("https://www.zhihu.com/pin/abc?x=1") == "https://www.zhihu.com/pin/abc"


def test_canonical_pin_url_unmatched_is_stripped() -> None:
    """Unparseable URLs are returned stripped."""
    assert canonical_pin_url("  https://example.com/foo  ") == "https://example.com/foo"


def test_split_composite_id() -> None:
    """Composite IDs yield the child part; bare IDs are unchanged."""
    assert split_composite_id("100/200") == "200"
    assert split_composite_id("200") == "200"
