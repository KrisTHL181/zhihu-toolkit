"""Tests for the collection download helpers (offline, no network)."""

from __future__ import annotations

from pathlib import Path

import click
import pytest

from zhihu_cli.commands.download import (
    _bucket_collection_items,
    _canonical_item_url,
    _collection_kind,
    _parse_collection_id,
    _pending_items,
)
from zhihu_cli.content.download_state import DownloadState


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("https://www.zhihu.com/collection/963088250", "963088250"),
        ("http://zhihu.com/collection/42", "42"),
        ("963088250", "963088250"),
        ("  123  ", "123"),
    ],
)
def test_parse_collection_id(ref: str, expected: str) -> None:
    """A collection URL or a bare numeric ID resolves to the numeric ID."""
    assert _parse_collection_id(ref) == expected


@pytest.mark.parametrize(
    "ref",
    [
        "https://www.zhihu.com/question/509876150",
        "not-a-collection",
        "",
        "12a",
    ],
)
def test_parse_collection_id_rejects_invalid(ref: str) -> None:
    """Anything that is neither a collection URL nor numeric is rejected."""
    with pytest.raises(click.BadParameter):
        _parse_collection_id(ref)


@pytest.mark.parametrize(
    ("item_type", "kind"),
    [("answer", "answers"), ("article", "articles"), ("pin", "pins")],
)
def test_collection_kind_maps_singular_types(item_type: str, kind: str) -> None:
    """The listing's singular types map to the manifest's plural kinds."""
    assert _collection_kind(item_type) == kind


def test_collection_kind_skips_unknown_types() -> None:
    """Unsupported item types have no manifest kind."""
    assert _collection_kind("zvideo") is None


@pytest.mark.parametrize(
    ("kind", "item_id", "url"),
    [
        ("answers", "3049502262", "https://www.zhihu.com/answer/3049502262"),
        ("articles", "12345", "https://zhuanlan.zhihu.com/p/12345"),
        ("pins", "99abc", "https://www.zhihu.com/pin/99abc"),
    ],
)
def test_canonical_item_url(kind: str, item_id: str, url: str) -> None:
    """Each kind gets the canonical URL shape DownloadState.bootstrap expects."""
    assert _canonical_item_url(kind, item_id) == url


def test_canonical_item_url_rejects_unknown_kind() -> None:
    """An unknown kind cannot produce a canonical URL."""
    with pytest.raises(ValueError):
        _canonical_item_url("zvideos", "1")


def test_bucket_collection_items_skips_unsupported() -> None:
    """Items are bucketed by type; unknown types and missing IDs are counted."""
    items = [
        {"type": "answer", "id": 1},
        {"type": "article", "id": 2},
        {"type": "pin", "id": "abc"},
        {"type": "zvideo", "id": 3},
        {"type": "answer"},
    ]

    buckets, unsupported = _bucket_collection_items(items)

    assert buckets["answers"] == [items[0]]
    assert buckets["articles"] == [items[1]]
    assert buckets["pins"] == [items[2]]
    assert unsupported == 2


def test_pending_items_respects_state_and_force(tmp_path: Path) -> None:
    """Fresh items are skipped unless --force is set."""
    answers_dir = tmp_path / "answers"
    answers_dir.mkdir()
    (answers_dir / "a.md").write_text("body", encoding="utf-8")

    state = DownloadState(str(tmp_path))
    state.load()
    state.record(
        "answers",
        "1",
        url="https://www.zhihu.com/answer/1",
        updated_time="2024-01-02 00:00:00",
        filepath="answers/a.md",
    )

    items = [
        {"id": 1, "updated_time": "2024-01-02 00:00:00"},
        {"id": 2, "updated_time": "2024-01-02 00:00:00"},
        {"id": 1, "updated_time": "2024-01-03 00:00:00"},
    ]

    pending = _pending_items(items, "answers", state, force=False)
    assert [item["id"] for item in pending] == [2, 1]
    assert [item["updated_time"] for item in pending] == ["2024-01-02 00:00:00", "2024-01-03 00:00:00"]

    assert _pending_items(items, "answers", state, force=True) == items
