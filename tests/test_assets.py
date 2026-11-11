"""Tests for the unified asset model and its per-source adapters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zhihu_cli.content.assets import (
    asset_from_item,
    asset_url,
    convert_files,
    convert_items,
    detect_format,
    load_assets,
    write_assets,
)


def test_detect_format_recognises_all_shapes() -> None:
    """Every supported source shape is detected from its characteristic keys."""
    assert detect_format([{"id": "1", "question": {"title": "Q"}}]) == "answers"
    assert detect_format([{"verb": "create", "target_type": "answer", "target_id": "1"}]) == "activities"
    assert detect_format([{"type": "answer", "data": {"id": "1"}}]) == "creations"
    assert detect_format([{"type": "article", "id": "1"}]) == "typed"
    assert detect_format([{"id": "1", "title": "T"}]) == "generic_title"


def test_detect_format_returns_unknown_and_skips_non_dicts() -> None:
    """Unrecognisable items (including leading non-dicts) fall through to unknown."""
    assert detect_format([None, "junk", {"foo": "bar"}]) == "unknown"
    assert detect_format([]) == "unknown"


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


def test_asset_from_activity_creation() -> None:
    """A creation activity becomes an asset using its target type and id."""
    item = {"verb": "create", "target_type": "pin", "target_id": 7, "title": "A pin"}

    assert asset_from_item(item, "activities") == {"id": "7", "type": "pin", "title": "A pin"}


def test_asset_from_activity_skips_non_creation() -> None:
    """Votes, follows and other non-creation activities are skipped."""
    item = {"verb": "vote", "target_type": "answer", "target_id": "7", "is_creation": False}

    assert asset_from_item(item, "activities") is None


def test_convert_items_answers_tolerates_bad_questions() -> None:
    """Null, missing, non-dict and null-title questions all become answers."""
    items = [
        {"id": "1", "question": None},
        {"id": "2"},
        {"id": "3", "question": "deleted"},
        {"id": "4", "question": {"title": None}},
        {"id": "5", "question": {"title": "First?"}},
    ]

    result = convert_items(items, source="answers")

    assert [asset["type"] for asset in result] == ["answer"] * 5
    assert [asset["title"] for asset in result] == [
        "untitled answer",
        "untitled answer",
        "untitled answer",
        "untitled answer",
        "First?",
    ]


def test_convert_items_answers_skips_null_id() -> None:
    """Answers whose id is ``None`` or empty are dropped, valid items are kept."""
    items = [
        {"id": None, "question": {"title": "dropped"}},
        {"id": "", "question": {"title": "dropped"}},
        {"id": "3", "question": {"title": "kept"}},
    ]

    assert convert_items(items, source="answers") == [{"id": "3", "type": "answer", "title": "kept"}]


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


def test_convert_files_detects_format_per_file(tmp_path: Path) -> None:
    """Mixed exports each use their own detected format, not the first file's."""
    answers = tmp_path / "answers.json"
    answers.write_text(json.dumps([{"id": "1", "question": {"title": "Q"}}]), encoding="utf-8")
    generic = tmp_path / "generic.json"
    generic.write_text(json.dumps([{"id": "2", "title": "Article"}]), encoding="utf-8")

    result = convert_files([answers, generic])

    assert {(asset["type"], asset["id"]) for asset in result} == {("answer", "1"), ("article", "2")}


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
