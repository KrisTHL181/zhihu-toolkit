"""Tests for the heuristic format detection in ``universal_converter``."""

from __future__ import annotations

from zhihu_cli.content.assets import convert_items as assets_convert_items
from zhihu_cli.content.assets import detect_format as assets_detect_format
from zhihu_cli.content.universal_converter import convert_items, detect_format


def test_shim_reexports_assets_helpers() -> None:
    """The deprecated shim exposes the very same functions as the assets module."""
    assert convert_items is assets_convert_items
    assert detect_format is assets_detect_format


def test_detect_format_answers() -> None:
    """A well-formed answer item is detected as the answers format."""
    assert detect_format([{"id": "1", "question": {"title": "First?"}}]) == "answers"


def test_detect_format_skips_leading_null_question() -> None:
    """A leading null question does not hide a later answer item."""
    items = [
        {"id": "2", "question": None},
        {"id": "1", "question": {"title": "First?"}},
    ]

    assert detect_format(items) == "answers"


def test_convert_items_mixed_shapes() -> None:
    """Null, missing, non-dict and null-title questions all become answers."""
    items = [
        {"id": "1", "question": None},
        {"id": "2"},
        {"id": "3", "question": "deleted"},
        {"id": "4", "question": {"title": None}},
        {"id": "5", "question": {"title": "First?"}},
    ]

    result = convert_items(items)

    assert [asset["type"] for asset in result] == ["answer"] * 5
    assert [asset["title"] for asset in result] == [
        "untitled answer",
        "untitled answer",
        "untitled answer",
        "untitled answer",
        "First?",
    ]


def test_convert_items_skips_null_id() -> None:
    """Items whose id is ``None`` or empty are dropped, valid items are kept."""
    items = [
        {"id": None, "question": {"title": "dropped"}},
        {"id": "", "question": {"title": "dropped"}},
        {"id": "3", "question": {"title": "kept"}},
    ]

    result = convert_items(items)

    assert result == [{"id": "3", "type": "answer", "title": "kept"}]


def test_convert_items_forced_type_keeps_answer_title() -> None:
    """A forced type no longer disables answer-title extraction."""
    result = convert_items([{"id": "1", "question": {"title": "First?"}}], forced_type="answer")

    assert result == [{"id": "1", "type": "answer", "title": "First?"}]
