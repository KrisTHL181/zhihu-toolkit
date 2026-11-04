"""Tests for the consecutive-empty-page guard in :mod:`zhihu_cli.content.handlers.waterfall`."""

from __future__ import annotations

from typing import Any

import pytest

from zhihu_cli.content.handlers import waterfall


class FakeResponse:
    """Minimal stand-in for a ``curl_cffi`` response."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def raise_for_status(self) -> None:
        """No-op: the fake never fails."""

    def json(self) -> dict[str, Any]:
        """Return the canned page body."""
        return self._data


class FakeSession:
    """Serve canned pages in order, repeating the last one if asked again."""

    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self.pages = pages
        self.calls: list[str] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        """Record the request and return the next canned page."""
        self.calls.append(url)
        return FakeResponse(self.pages[min(len(self.calls) - 1, len(self.pages) - 1)])


def _page(items: list[Any], next_url: str | None = None, is_end: bool = False) -> dict[str, Any]:
    """Build one API page body."""
    paging: dict[str, Any] = {"is_end": is_end}
    if next_url is not None:
        paging["next"] = next_url
    return {"data": items, "paging": paging}


def _run(
    monkeypatch: pytest.MonkeyPatch,
    pages: list[dict[str, Any]],
    max_items: int | None = None,
) -> tuple[waterfall.PageStream, list[Any], FakeSession]:
    """Drive ``stream_handler`` over ``pages`` and return the stream, items and fake session."""
    fake = FakeSession(pages)
    monkeypatch.setattr(waterfall, "session", fake)
    monkeypatch.setattr(waterfall, "wait", lambda *args, **kwargs: None)
    stream = waterfall.stream_handler(
        "https://api.zhihu.com/topic/1/followers?offset=0",
        lambda data: data.get("data", []),
        delay=1.0,
        max_items=max_items,
    )
    return stream, list(stream), fake


def test_three_consecutive_empty_pages_stall(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three empty pages in a row end pagination without following ``next`` further."""
    pages = [
        _page([1, 2], next_url="https://api.zhihu.com/topic/1/followers?offset=20"),
        _page([], next_url="https://api.zhihu.com/topic/1/followers?offset=40"),
        _page([], next_url="https://api.zhihu.com/topic/1/followers?offset=60"),
        _page([], next_url="https://api.zhihu.com/topic/1/followers?offset=80"),
    ]

    stream, items, fake = _run(monkeypatch, pages)

    assert items == [1, 2]
    assert stream.stalled is True
    assert len(fake.calls) == 4  # one data page + exactly three empty pages


def test_empty_pages_then_data_continue(monkeypatch: pytest.MonkeyPatch) -> None:
    """A page with items resets the consecutive-empty counter."""
    pages = [
        _page([1, 2], next_url="https://api.zhihu.com/topic/1/followers?offset=20"),
        _page([], next_url="https://api.zhihu.com/topic/1/followers?offset=40"),
        _page([], next_url="https://api.zhihu.com/topic/1/followers?offset=60"),
        _page([3, 4], next_url="https://api.zhihu.com/topic/1/followers?offset=80"),
        _page([], is_end=True),
    ]

    stream, items, _ = _run(monkeypatch, pages)

    assert items == [1, 2, 3, 4]
    assert stream.stalled is False


def test_is_end_after_one_empty_page_ends_normally(monkeypatch: pytest.MonkeyPatch) -> None:
    """``is_end=true`` still ends the walk naturally after a single empty page."""
    pages = [
        _page([1], next_url="https://api.zhihu.com/topic/1/followers?offset=20"),
        _page([], next_url="https://api.zhihu.com/topic/1/followers?offset=40"),
        _page([], is_end=True),
    ]

    stream, items, _ = _run(monkeypatch, pages)

    assert items == [1]
    assert stream.stalled is False


def test_max_items_truncation_is_not_a_stall(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stopping mid-page on ``max_items`` sets ``truncated`` and not ``stalled``."""
    pages = [_page([1, 2, 3, 4, 5], next_url="https://api.zhihu.com/topic/1/followers?offset=20")]

    stream, items, fake = _run(monkeypatch, pages, max_items=2)

    assert items == [1, 2]
    assert stream.truncated is True
    assert stream.stalled is False
    assert len(fake.calls) == 1
