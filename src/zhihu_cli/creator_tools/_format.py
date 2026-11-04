"""Shared value formatting helpers for the creator tools.

The module is stdlib-only so the creator tools can use it even when the
optional analytics stack is not installed.
"""

from __future__ import annotations

__all__ = ["fmt_num", "ratio_bar"]


def fmt_num(value: float | int, is_int: bool = True) -> str:
    """Format a number for display.

    :param value: The numeric value.
    :param is_int: When true, render as an integer with comma grouping;
        otherwise render with two decimal places.
    :returns: The formatted string.
    """
    if is_int:
        return f"{int(value):,}"
    return f"{value:.2f}"


def ratio_bar(score: float, max_val: float, width: int = 14) -> str:
    """Render a unicode bar showing the ``score / max_val`` ratio.

    :param score: The value to show.
    :param max_val: The scale maximum. Zero yields an empty string.
    :param width: Number of bar cells.
    :returns: The bar followed by the clamped percentage.
    """
    if max_val == 0:
        return ""
    pct = min(score / max_val, 1.0)
    filled = int(pct * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar} {pct * 100:.0f}%"
