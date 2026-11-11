"""Tests for the full-content member-answer stream in ``handlers.people``."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from zhihu_cli.content.handlers import fmt_time, people


def _item(**overrides: Any) -> dict[str, Any]:
    """Build one raw member-answer item with every field the full mode reads."""
    item: dict[str, Any] = {
        "id": 101,
        "question": {"id": 42, "title": "Question title", "created": 1_600_000_000},
        "author": {"name": "Alice", "url_token": "alice"},
        "content": "<p>hello</p>",
        "excerpt": "hello",
        "voteup_count": 7,
        "comment_count": 2,
        "favlists_count": 3,
        "thanks_count": 4,
        "created_time": 1_700_000_000,
        "updated_time": 1_700_000_100,
        "content_need_truncated": False,
        "is_copyable": True,
    }
    item.update(overrides)
    return item


def _patch_stream(monkeypatch: pytest.MonkeyPatch, items: list[dict[str, Any]]) -> dict[str, Any]:
    """Replace ``people.stream_handler`` and capture the URL and parsed items."""
    captured: dict[str, Any] = {}

    def fake_stream(url: str, parser: Any, **kwargs: Any) -> Any:
        captured["url"] = url
        captured["kwargs"] = kwargs
        captured["items"] = list(parser({"data": items}))
        return iter(captured["items"])

    monkeypatch.setattr(people, "stream_handler", fake_stream)
    return captured


def test_stream_full_content_requests_body_and_vote_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """``include_content`` asks for ``content`` and honours ``sort_by``."""
    captured = _patch_stream(monkeypatch, [_item()])

    result = list(people.stream_member_answers("alice", include_content=True, sort_by="voteups", raw=True))

    assert "sort_by=voteups" in captured["url"]
    assert "data%5B%2A%5D.content" in captured["url"]
    assert captured["kwargs"]["max_items"] is None
    assert captured["kwargs"]["max_pages"] is None
    assert captured["kwargs"]["delay"] == 1.0
    assert result[0]["content"] == "<p>hello</p>"
    assert result[0]["author"] == "Alice"
    assert result[0]["author_token"] == "alice"
    assert result[0]["vote"] == 7
    assert result[0]["comment"] == 2
    assert result[0]["favorite"] == 3
    assert result[0]["thanks"] == 4
    assert result[0]["created_time"] == 1_700_000_000
    assert result[0]["question_created"] == 1_600_000_000
    assert result[0]["truncated"] is False


def test_stream_full_content_keeps_unknown_favorite_as_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """The author path omits favlists_count; ``None`` means "unknown", not zero."""
    item = _item()
    del item["favlists_count"]
    _patch_stream(monkeypatch, [item])

    result = list(people.stream_member_answers("alice", include_content=True, raw=True))

    assert result[0]["favorite"] is None


def test_stream_full_content_converts_to_markdown_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``raw`` the answer body is converted to Markdown."""
    _patch_stream(monkeypatch, [_item()])

    result = list(people.stream_member_answers("alice", include_content=True))

    assert "<p>" not in result[0]["content"]
    assert "hello" in result[0]["content"]


def test_stream_listing_shape_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default mode keeps the historical excerpt/formatted-time shape."""
    captured = _patch_stream(monkeypatch, [_item()])

    result = list(people.stream_member_answers("alice"))

    assert "sort_by=created" in captured["url"]
    assert "data%5B%2A%5D.excerpt" in captured["url"]
    assert set(result[0]) == {
        "type",
        "id",
        "question_id",
        "title",
        "excerpt",
        "url",
        "created_time",
        "updated_time",
        "voteup_count",
        "comment_count",
        "is_copyable",
        "question_created",
    }
    assert result[0]["created_time"] == fmt_time(1_700_000_000)
    assert result[0]["question_created"] == 1_600_000_000


def test_fetch_member_answers_forwards_new_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """The eager wrapper is the list form of the same stream."""
    _patch_stream(monkeypatch, [_item()])

    result = people.fetch_member_answers("alice", include_content=True, sort_by="voteups", raw=True)

    assert len(result) == 1
    assert result[0]["content"] == "<p>hello</p>"


def test_since_requires_created_order() -> None:
    """``since`` cannot stop a vote-ordered walk without dropping items."""
    with pytest.raises(ValueError, match="newest-first"):
        people.stream_member_answers("alice", since="2024-01-01", sort_by="voteups")


def test_unknown_sort_order_is_rejected() -> None:
    """A typo in ``sort_by`` fails fast instead of silently using the wrong order."""
    with pytest.raises(ValueError, match="sort_by must be one of"):
        people.stream_member_answers("alice", sort_by="votes")


def test_until_since_handles_integer_timestamps() -> None:
    """The full-content shape uses epoch integers; the cutoff still works."""
    new = int(datetime(2024, 1, 1).timestamp())
    old = int(datetime(2023, 1, 1).timestamp())

    result = list(people._until_since([{"id": 1, "created_time": new}, {"id": 2, "created_time": old}], "2023-06-01"))

    assert [item["id"] for item in result] == [1]
