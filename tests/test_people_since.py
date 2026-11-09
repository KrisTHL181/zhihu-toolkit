"""Tests for the incremental listing helper in ``handlers.people``."""

from __future__ import annotations

from zhihu_cli.content.handlers.people import _until_since


def _item(created: str, item_id: int) -> dict:
    """Build a minimal parsed listing item."""
    return {"id": item_id, "created_time": created}


def test_until_since_stops_at_first_older_item() -> None:
    """Descending input stops before the first item older than *since*."""
    items = [
        _item("2024-03-01 12:00:00", 1),
        _item("2024-02-20 12:00:00", 2),
        _item("2024-02-10 12:00:00", 3),
        _item("2024-01-01 12:00:00", 4),
    ]

    result = list(_until_since(items, "2024-02-15"))

    assert [item["id"] for item in result] == [1, 2]


def test_until_since_none_yields_everything() -> None:
    """Without a bound the stream is passed through unchanged."""
    items = [_item("2024-01-01 00:00:00", 1), _item("2023-01-01 00:00:00", 2)]

    assert list(_until_since(items, None)) == items


def test_until_since_does_not_assume_ascending_order() -> None:
    """Ascending input is yielded as-is; the helper never sorts or rescans."""
    items = [_item("2024-01-01 00:00:00", 1), _item("2024-02-01 00:00:00", 2)]

    assert list(_until_since(items, "2023-01-01")) == items
    assert list(_until_since(items, "2024-03-01")) == []


def test_until_since_keeps_unknown_timestamps() -> None:
    """Items without a parseable date do not trigger the stop condition."""
    items = [_item("unknown time", 1), _item("2024-03-01 00:00:00", 2)]

    assert [item["id"] for item in _until_since(items, "2024-02-01")] == [1, 2]
