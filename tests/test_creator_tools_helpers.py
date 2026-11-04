"""Tests for the shared creator-tool helpers."""

from __future__ import annotations

import pytest

from zhihu_cli.creator_tools._format import fmt_num, ratio_bar
from zhihu_cli.creator_tools._io import load_json


def test_load_json_reads_file(tmp_path) -> None:
    """Valid JSON is returned as-is."""
    path = tmp_path / "data.json"
    path.write_text('{"a": 1}', encoding="utf-8")

    assert load_json(path) == {"a": 1}


def test_load_json_returns_default_on_missing_file(tmp_path) -> None:
    """A missing file yields the default when one is supplied."""
    assert load_json(tmp_path / "missing.json", default=None) is None


def test_load_json_returns_default_on_bad_json(tmp_path) -> None:
    """Malformed JSON yields the default when one is supplied."""
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")

    assert load_json(path, default=[]) == []


def test_load_json_reraises_without_default(tmp_path) -> None:
    """Without a default the original exception propagates."""
    with pytest.raises(FileNotFoundError):
        load_json(tmp_path / "missing.json")


def test_fmt_num_integer_and_float() -> None:
    """Integers get grouping, floats get two decimals."""
    assert fmt_num(1234567) == "1,234,567"
    assert fmt_num(1234.5, is_int=False) == "1234.50"
    assert fmt_num(1234.5, False) == "1234.50"


def test_ratio_bar_zero_max() -> None:
    """A zero maximum renders nothing."""
    assert ratio_bar(0, 0) == ""


def test_ratio_bar_clamps_over_range() -> None:
    """Values above the maximum fill the bar completely."""
    assert ratio_bar(200, 100, width=10) == "██████████ 100%"


def test_ratio_bar_partial() -> None:
    """A partial ratio fills proportionally and rounds the percentage."""
    assert ratio_bar(50, 100, width=10) == "█████░░░░░ 50%"
