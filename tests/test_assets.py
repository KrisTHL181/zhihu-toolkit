"""Tests for the unified asset model and its per-source adapters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zhihu_cli.content.assets import (
    asset_from_item,
    asset_url,
    load_assets,
    write_assets,
)


def test_asset_from_creation() -> None:
    """A creator creation yields its id, type, title and created_time."""
    item = {"type": "article", "data": {"id": 123, "title": "Hello", "created_time": 42}}

    assert asset_from_item(item, "creations") == {
        "id": "123",
        "type": "article",
        "title": "Hello",
        "created_time": 42,
    }


@pytest.mark.parametrize(
    "item",
    [
        {"type": "zvideo", "data": {"id": "1"}},
        {"type": "answer", "data": {"title": "no id"}},
        {"type": "answer", "data": "not a dict"},
        {"type": "answer"},
    ],
)
def test_asset_from_creation_rejects_unsupported_items(item: dict) -> None:
    """Unsupported types and malformed/missing ``data`` are rejected."""
    assert asset_from_item(item, "creations") is None


def test_asset_from_article() -> None:
    """A raw article item becomes an article asset keyed by its id."""
    item = {"id": 1, "title": "Raw article"}

    assert asset_from_item(item, "article") == {"id": "1", "type": "article", "title": "Raw article"}


def test_asset_from_article_rejects_missing_id() -> None:
    """An article item without an id cannot become an asset."""
    assert asset_from_item({"title": "no id"}, "article") is None


def test_load_assets_skips_malformed_entries(tmp_path: Path) -> None:
    """Entries without an id/type or that are not dicts are skipped."""
    source = tmp_path / "assets.json"
    source.write_text(
        json.dumps(
            [
                {"id": 1, "type": "answer", "title": "A"},
                {"type": "article", "title": "no id"},
                {"id": 2, "title": "no type"},
                "not a dict",
                {"id": "", "type": "pin"},
                {"id": 3, "type": 4},
                {"id": "4", "type": "pin", "title": "P", "url": "https://example.com", "created_time": 5},
            ]
        ),
        encoding="utf-8",
    )

    assets = load_assets(source)

    assert assets == [
        {"id": "1", "type": "answer", "title": "A"},
        {"id": "3", "type": "4", "title": ""},
        {"id": "4", "type": "pin", "title": "P", "url": "https://example.com", "created_time": 5},
    ]


def test_write_assets_creates_parents_and_round_trips(tmp_path: Path) -> None:
    """write_assets creates missing directories and its output reloads intact."""
    target = tmp_path / "nested" / "dir" / "assets.json"
    assets = [{"id": "1", "type": "answer", "title": "A"}, {"id": "2", "type": "pin", "title": "P"}]

    write_assets(target, assets)

    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8")) == assets
    assert load_assets(target) == assets


@pytest.mark.parametrize(
    ("asset_type", "asset_id", "url"),
    [
        ("answer", "3049502262", "https://www.zhihu.com/answer/3049502262"),
        ("article", "12345", "https://zhuanlan.zhihu.com/p/12345"),
        ("pin", "99abc", "https://www.zhihu.com/pin/99abc"),
    ],
)
def test_asset_url(asset_type: str, asset_id: str, url: str) -> None:
    """Each asset type gets its canonical URL shape."""
    assert asset_url({"id": asset_id, "type": asset_type, "title": ""}) == url


def test_asset_url_rejects_unknown_type() -> None:
    """Unknown asset types cannot produce a canonical URL."""
    with pytest.raises(ValueError):
        asset_url({"id": "1", "type": "zvideo", "title": ""})
