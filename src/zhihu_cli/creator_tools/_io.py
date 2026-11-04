"""Shared JSON loading helper for the creator tools.

The module is stdlib-only so the creator tools can use it even when the
optional analytics stack is not installed. Error messages and warnings stay in
the call sites; this helper only centralises the open/parse step.
"""

from __future__ import annotations

import json
import os
from typing import Any

__all__ = ["load_json"]

_MISSING = object()


def load_json(
    path: str | os.PathLike[str],
    default: Any = _MISSING,
    *,
    encoding: str = "utf-8",
) -> Any:
    """Load a JSON file.

    :param path: File to read.
    :param default: Value returned when the file cannot be read or parsed. When
        omitted, the original exception is re-raised.
    :param encoding: Text encoding used to read the file.
    :returns: The parsed JSON value, or ``default`` when given and loading
        failed.
    :raises OSError: When ``default`` is not given and the file cannot be read.
    :raises json.JSONDecodeError: When ``default`` is not given and the content
        is not valid JSON.
    """
    try:
        with open(path, encoding=encoding) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        if default is _MISSING:
            raise
        return default
