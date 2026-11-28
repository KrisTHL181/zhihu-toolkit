"""Tests for the comment-review handlers (``fetch_reviewing_comments`` / ``approve_comment``)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from zhihu_cli.content.handlers import comments


def _patch_stream(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace ``comments.stream_handler`` and capture the URL, kwargs, and parser."""
    captured: dict[str, Any] = {}

    def fake_stream(url: str, parser: Any, **kwargs: Any) -> Any:
        captured["url"] = url
        captured["kwargs"] = kwargs
        captured["parser"] = parser
        return iter([])

    monkeypatch.setattr(comments, "stream_handler", fake_stream)
    return captured


def test_list_url_and_normalisation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The list call hits the review endpoint and normalises each comment."""
    captured = _patch_stream(monkeypatch)

    result = list(comments.fetch_reviewing_comments("articles", "123", limit=5))

    assert captured["url"].startswith("https://api.zhihu.com/comment_v5/articles/123/reviewing_comments?order_by=score")
    assert captured["kwargs"]["limit"] == 5
    assert result == []

    page = {
        "counts": {"total_counts": 1},
        "data": [
            {
                "id": 7001,
                "content": "<p>hi</p>",
                "created_time": 1_700_000_000,
                "author": {"name": "Alice", "id": "a-1", "url_token": "alice"},
                "reply_comment_id": "6999",
                "can_delete": True,
                "reviewing": True,
            }
        ],
        "paging": {"is_end": True, "next": "", "previous": "", "totals": 1},
    }

    items = list(captured["parser"](page))
    assert len(items) == 1

    content = items[0].pop("content")
    assert "<p>" not in content
    assert "hi" in content
    assert items[0] == {
        "id": 7001,
        "author": "Alice",
        "author_id": "a-1",
        "author_token": "alice",
        "created_time": 1_700_000_000,
        "reply_comment_id": "6999",
        "can_delete": True,
        "reviewing": True,
    }


def test_plural_and_singular_item_types(monkeypatch: pytest.MonkeyPatch) -> None:
    """Singular item types are mapped to the plural URL segment."""
    captured = _patch_stream(monkeypatch)

    list(comments.fetch_reviewing_comments("answers", "456"))
    plural_url = captured["url"]

    list(comments.fetch_reviewing_comments("answer", "456"))
    singular_url = captured["url"]

    assert "/answers/456/reviewing_comments" in plural_url
    assert singular_url == plural_url


def test_approve_comment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Approval PUTs the operation endpoint and requires ``success``."""
    calls: list[str] = []

    class FakeResponse:
        text = '{"success": true}'
        ok = True
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {"success": True}

    class FakeSession:
        def put(self, url: str) -> FakeResponse:
            calls.append(url)
            return FakeResponse()

    monkeypatch.setattr(comments, "session", FakeSession())

    assert comments.approve_comment("42") == {"success": True}
    assert calls == ["https://www.zhihu.com/api/v4/comment_v5/comment/42/operation/approve"]


def test_approve_comment_rejects_unsuccessful_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """A response without ``success: true`` raises ``RuntimeError``."""
    failure = SimpleNamespace(
        text='{"success": false}',
        ok=True,
        status_code=200,
        json=lambda: {"success": False},
    )
    monkeypatch.setattr(comments, "session", SimpleNamespace(put=lambda url: failure))

    with pytest.raises(RuntimeError, match="Failed to approve comment 42"):
        comments.approve_comment("42")
