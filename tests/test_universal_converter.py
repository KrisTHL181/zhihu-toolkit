"""Tests for the universal-converter item handling (no network access)."""

from __future__ import annotations

from zhihu_cli.content.universal_converter import convert_items, detect_format


def test_detect_format_answers() -> None:
    items = [{"id": "1", "question": {"title": "First?"}}]

    assert detect_format(items) == "answers"


def test_convert_items_answers_format_with_null_question() -> None:
    """A null/missing ``question`` must not crash title extraction."""
    items = [
        {"id": "1", "question": {"title": "First?"}},
        {"id": "2", "question": None},
        {"id": "3"},
    ]

    assets = convert_items(items)

    assert [a["title"] for a in assets] == [
        "First?",
        "untitled answer",
        "untitled answer",
    ]
    assert all(a["type"] == "answer" for a in assets)
