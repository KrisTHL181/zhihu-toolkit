"""Tests for :meth:`zhihu_cli.content.handlers.waterfall.PageStream.close`."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
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


def _page(
    items: list[Any],
    *,
    next_url: str | None = None,
    is_end: bool = False,
    totals: int | None = None,
) -> dict[str, Any]:
    """Build one API page body."""
    paging: dict[str, Any] = {"is_end": is_end}
    if next_url is not None:
        paging["next"] = next_url
    if totals is not None:
        paging["totals"] = totals
    return {"data": items, "paging": paging}


def _stream(
    monkeypatch: pytest.MonkeyPatch,
    pages: list[dict[str, Any]],
) -> tuple[waterfall.PageStream, FakeSession]:
    """Build a stream over ``pages`` without network or delay."""
    fake = FakeSession(pages)
    monkeypatch.setattr(waterfall, "session", fake)
    monkeypatch.setattr(waterfall, "wait", lambda *args, **kwargs: None)
    stream = waterfall.stream_handler(
        "https://api.zhihu.com/topic/1/followers?offset=0",
        lambda data: data.get("data", []),
    )
    return stream, fake


def test_close_stops_iteration_without_fetching_another_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """Closing after one item ends the loop and never requests page two."""
    pages = [
        _page([1, 2], next_url="https://api.zhihu.com/topic/1/followers?offset=20"),
        _page([3], is_end=True),
    ]
    stream, fake = _stream(monkeypatch, pages)

    assert next(stream) == 1
    stream.close()

    assert len(fake.calls) == 1
    assert list(stream) == []
    with pytest.raises(StopIteration):
        next(stream)


def test_close_before_first_item_does_not_send_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stream closed before iteration never touches the network."""
    stream, fake = _stream(monkeypatch, [_page([1], is_end=True)])

    stream.close()

    assert fake.calls == []
    assert list(stream) == []


def test_close_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling close twice neither fails nor fetches another page."""
    pages = [
        _page([1, 2], next_url="https://api.zhihu.com/topic/1/followers?offset=20"),
        _page([3], is_end=True),
    ]
    stream, fake = _stream(monkeypatch, pages)

    assert next(stream) == 1
    stream.close()
    stream.close()

    assert len(fake.calls) == 1
    assert list(stream) == []


def test_close_after_exhaustion_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Closing a naturally exhausted stream is harmless."""
    stream, fake = _stream(monkeypatch, [_page([1, 2], is_end=True)])

    assert list(stream) == [1, 2]
    calls_before = len(fake.calls)
    stream.close()

    assert len(fake.calls) == calls_before
    assert stream.stalled is False
    assert stream.truncated is False


def test_close_closes_the_backing_generator() -> None:
    """``close()`` runs the source generator's cleanup immediately."""
    closed: list[bool] = []

    def source() -> Iterator[int]:
        try:
            yield 1
            yield 2
        finally:
            closed.append(True)

    stream = waterfall.PageStream(source(), waterfall._StreamStats())

    assert next(stream) == 1
    assert closed == []
    stream.close()
    assert closed == [True]
    with pytest.raises(StopIteration):
        next(stream)


def test_close_works_when_source_has_no_close() -> None:
    """A plain iterator without ``close()`` still becomes exhausted."""
    stream = waterfall.PageStream(iter([1, 2]), waterfall._StreamStats())

    assert next(stream) == 1
    stream.close()
    assert list(stream) == []


def test_close_leaves_stalled_and_truncated_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """A consumer abort is neither a server stall nor an intentional cap."""
    pages = [
        _page([1, 2], next_url="https://api.zhihu.com/topic/1/followers?offset=20", totals=10),
        _page([3], is_end=True),
    ]
    stream, _ = _stream(monkeypatch, pages)

    assert next(stream) == 1
    stream.close()

    assert stream.yielded == 1
    assert stream.totals == 10
    assert stream.stalled is False
    assert stream.truncated is False


def test_close_does_not_emit_incomplete_stream_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """The missing-tail warning is for natural ends, not for a deliberate cancel."""
    pages = [
        _page([1, 2], next_url="https://api.zhihu.com/topic/1/followers?offset=20", totals=10),
        _page([3], is_end=True),
    ]
    stream, _ = _stream(monkeypatch, pages)

    assert next(stream) == 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stream.close()

    assert caught == []
