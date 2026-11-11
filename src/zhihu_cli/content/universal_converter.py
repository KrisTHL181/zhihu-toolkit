"""Deprecated compatibility shim for the unified asset converter.

The implementation moved to :mod:`zhihu_cli.content.assets`. New code should
import :func:`detect_format`, :func:`convert_items`, and :func:`load_json`
from there instead. This module remains importable for extensions and older
scripts.
"""

from __future__ import annotations

from zhihu_cli.content.assets import convert_items, detect_format, load_json

__all__ = ["convert_items", "detect_format", "load_json"]
