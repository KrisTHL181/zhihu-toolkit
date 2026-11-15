"""Tests for the interact vote/like dispatch — no network calls.

The handler symbols are monkeypatched on :mod:`zhihu_cli.commands.interact`
itself, because both :func:`_dispatch_vote` and the Click commands resolve
them from that module's globals at call time.
"""

from __future__ import annotations

from typing import Any

import click
import pytest
from click.testing import CliRunner

from zhihu_cli.commands import interact

_HANDLERS = (
    "upvote_answer",
    "downvote_answer",
    "neutral_answer",
    "upvote_question",
    "downvote_question",
    "upvote_article",
    "downvote_article",
    "neutral_article",
    "like_article",
    "unlike_article",
    "upvote_pin",
    "neutral_pin",
)


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    """Replace every vote/like handler with a recorder and return its log."""
    recorded: list[tuple[Any, ...]] = []

    def recorder(name: str):
        def fake(*args: Any) -> dict[str, Any]:
            recorded.append((name, *args))
            return {"success": True, "handler": name}

        return fake

    for name in _HANDLERS:
        monkeypatch.setattr(interact, name, recorder(name))
    return recorded


def _cli() -> click.Group:
    """Build a root group carrying the registered interact commands."""

    @click.group()
    def root() -> None:
        pass

    interact.register_interact(root)
    return root


def _run(*args: str) -> Any:
    """Invoke ``zhihu interact ...`` and assert it succeeded."""
    result = CliRunner().invoke(_cli(), ["interact", *args])
    assert result.exit_code == 0, result.output
    return result


def test_answer_url_resolves_composite_id(calls: list[tuple[Any, ...]]) -> None:
    """The composite ``question_id/answer_id`` form votes on the answer part."""
    _run("vote", "up", "https://www.zhihu.com/question/111/answer/222")

    assert calls == [("upvote_answer", "222")]


def test_question_url_upvotes_question(calls: list[tuple[Any, ...]]) -> None:
    _run("vote", "up", "https://www.zhihu.com/question/111")

    assert calls == [("upvote_question", "111")]


def test_article_url_upvotes_article(calls: list[tuple[Any, ...]]) -> None:
    _run("vote", "up", "https://zhuanlan.zhihu.com/p/333")

    assert calls == [("upvote_article", "333")]


def test_pin_url_upvotes_pin(calls: list[tuple[Any, ...]]) -> None:
    _run("vote", "up", "https://www.zhihu.com/pin/444")

    assert calls == [("upvote_pin", "444")]


def test_article_url_downvotes_article(calls: list[tuple[Any, ...]]) -> None:
    _run("vote", "down", "https://zhuanlan.zhihu.com/p/333")

    assert calls == [("downvote_article", "333")]


def test_downvote_answer_and_question(calls: list[tuple[Any, ...]]) -> None:
    _run("vote", "down", "https://www.zhihu.com/question/111/answer/222")
    _run("vote", "down", "https://www.zhihu.com/question/111")

    assert calls == [("downvote_answer", "222"), ("downvote_question", "111")]


def test_pin_downvote_is_rejected(calls: list[tuple[Any, ...]]) -> None:
    """Pins have no downvote, so the dispatch raises instead of calling out."""
    with pytest.raises(click.ClickException, match="Pins do not support downvotes"):
        interact._dispatch_vote("pins", "444", "down")

    assert calls == []


def test_neutral_article_and_pin(calls: list[tuple[Any, ...]]) -> None:
    _run("vote", "neutral", "https://zhuanlan.zhihu.com/p/333")
    _run("vote", "neutral", "https://www.zhihu.com/pin/444")

    assert calls == [("neutral_article", "333"), ("neutral_pin", "444")]


def test_neutral_answer_and_question(calls: list[tuple[Any, ...]]) -> None:
    """Questions have always been cleared through ``neutral_answer``."""
    _run("vote", "neutral", "https://www.zhihu.com/question/111/answer/222")
    _run("vote", "neutral", "https://www.zhihu.com/question/111")

    assert calls == [("neutral_answer", "222"), ("neutral_answer", "111")]


def test_bare_raw_id_falls_back_to_answers(calls: list[tuple[Any, ...]]) -> None:
    """A raw ID carries no type and keeps the historical answer fallback."""
    interact._dispatch_vote(None, "999", "up")
    interact._dispatch_vote(None, "999", "down")
    interact._dispatch_vote(None, "999", "neutral")

    assert calls == [("upvote_answer", "999"), ("downvote_answer", "999"), ("neutral_answer", "999")]


def test_like_add_and_remove(calls: list[tuple[Any, ...]]) -> None:
    _run("like", "add", "https://zhuanlan.zhihu.com/p/333")
    _run("like", "remove", "https://zhuanlan.zhihu.com/p/333")

    assert calls == [("like_article", "333"), ("unlike_article", "333")]


def test_like_accepts_a_raw_article_id(calls: list[tuple[Any, ...]]) -> None:
    _run("like", "add", "333")

    assert calls == [("like_article", "333")]


def test_like_rejects_non_article_types(calls: list[tuple[Any, ...]]) -> None:
    result = CliRunner().invoke(_cli(), ["interact", "like", "add", "https://www.zhihu.com/question/111"])

    assert result.exit_code != 0
    assert "Likes are only supported for articles." in result.output
    assert calls == []
