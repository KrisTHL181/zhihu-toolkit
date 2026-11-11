"""Tests for ``handlers.question.stream_answers`` (order/include/question metadata)."""

from __future__ import annotations

from typing import Any
from urllib.parse import unquote

import pytest

from zhihu_cli.content.handlers import question


def _answer(**overrides: Any) -> dict[str, Any]:
    """Build one raw answer-feed item, wrapped the way ``/feeds`` returns it."""
    answer: dict[str, Any] = {
        "id": 99,
        "content": "<p>body</p>",
        "voteup_count": 5,
        "comment_count": 1,
        "favlists_count": 2,
        "thanks_count": 3,
        "created_time": 1_700_000_000,
        "updated_time": 1_700_000_100,
        "author": {"name": "Bob", "url_token": "bob"},
        "question": {"id": 42, "title": "Question title", "created": 1_600_000_000, "detail": "<p>desc</p>"},
        "content_need_truncated": True,
        "is_collapsed": False,
        "collapse_reason": "",
        "collapsed_by": "",
    }
    answer.update(overrides)
    return {"target": answer}


def _patch_stream(monkeypatch: pytest.MonkeyPatch, items: list[dict[str, Any]]) -> dict[str, Any]:
    """Replace ``question.stream_handler`` and capture the URL and parsed items."""
    captured: dict[str, Any] = {}

    def fake_stream(url: str, parser: Any, **kwargs: Any) -> Any:
        captured["url"] = url
        captured["kwargs"] = kwargs
        captured["items"] = list(parser({"data": items}))
        return iter(captured["items"])

    monkeypatch.setattr(question, "stream_handler", fake_stream)
    return captured


def test_order_and_question_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """``order=created`` + ``include_question`` exposes graph/corpus metadata."""
    captured = _patch_stream(monkeypatch, [_answer()])

    result = list(question.stream_answers({"id": 42}, raw=True, order="created", include_question=True))

    url = unquote(captured["url"])
    assert "order=created" in url
    assert "question.detail" in url
    assert captured["kwargs"] == {"limit": 5, "max_items": None}
    answer = result[0]
    assert answer["content"] == "<p>body</p>"
    assert answer["question_id"] == "42"
    assert answer["question_title"] == "Question title"
    assert answer["question_created"] == 1_600_000_000
    assert answer["question_description"] == "<p>desc</p>"
    assert answer["author_token"] == "bob"
    assert answer["thanks"] == 3
    assert answer["truncated"] is True


def test_question_description_is_converted_to_markdown_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``raw`` flag governs both the answer body and the question detail."""
    _patch_stream(monkeypatch, [_answer()])

    result = list(question.stream_answers({"id": 42}, include_question=True))

    assert "<p>" not in result[0]["content"]
    assert "<p>" not in result[0]["question_description"]
    assert "desc" in result[0]["question_description"]


def test_default_shape_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing callers see the same keys and the default order/include."""
    captured = _patch_stream(monkeypatch, [_answer()])

    result = list(question.stream_answers({"id": 42}, raw=True))

    url = unquote(captured["url"])
    assert "order=default" in url
    assert "question.detail" not in url
    assert "question_id" not in result[0]
    assert set(result[0]) == {
        "author",
        "id",
        "vote",
        "comment",
        "favorite",
        "created_time",
        "updated_time",
        "content",
        "is_collapsed",
        "collapse_reason",
        "collapsed_by",
    }


def test_include_overrides_the_whole_field_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit ``include`` replaces the default list."""
    captured = _patch_stream(monkeypatch, [_answer()])

    list(question.stream_answers({"id": 42}, include="content,question.detail", raw=True))

    url = unquote(captured["url"])
    assert "include=content,question.detail" in url
    assert "big_card_summary" not in url


def test_scrape_answers_remains_a_positional_compatible_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    """The old entry point forwards its positional arguments unchanged."""
    captured = _patch_stream(monkeypatch, [_answer()])

    question.scrape_answers({"id": 42}, True, 3, 10, False)

    assert captured["kwargs"] == {"limit": 3, "max_items": 10}
    assert captured["items"][0]["content"] == "<p>body</p>"


def test_empty_order_is_rejected() -> None:
    """An empty order would produce an invalid URL; fail before the request."""
    with pytest.raises(ValueError, match="order must be a non-empty string"):
        question.stream_answers({"id": 42}, order="")
